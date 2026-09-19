from types import SimpleNamespace

from src.core.context_manager import ContextManager


class FakeProvider:
    def __init__(self, token_scale=1):
        self.llm_config = SimpleNamespace(model="deepseek-flash", max_tokens=4096, api_key="test")
        self.token_scale = token_scale
        self.summary_calls = 0
        self.summary_prompts = []
        self.fail_summary = False

    def count_tokens(self, text):
        return len(text) // self.token_scale

    def context_window_for_model(self, model):
        return 1_000_000 if "flash" in model else 32_768

    def input_token_budget(self):
        return 994_904

    def request_output_cap(self):
        return 4_096

    def chat_completion(self, **kwargs):
        self.summary_calls += 1
        self.summary_prompts.append(kwargs["user_prompt"])
        if self.fail_summary:
            raise RuntimeError("offline")
        return "保留用户目标、文章主题和修改约束。"


def messages(count, width=20):
    return [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn-{index} " + "x" * width}
        for index in range(count)
    ]


def test_estimate_uses_configured_model_window_and_soft_limit():
    provider = FakeProvider()
    manager = ContextManager(provider)

    snapshot = manager.estimate(messages(2), documents=[{"content": "document"}], profile_text="profile")

    assert snapshot.context_window == 1_000_000
    assert snapshot.input_budget == 994_904
    assert snapshot.soft_limit == 32_000
    assert snapshot.output_cap == 4_096
    assert snapshot.used_tokens > 0


def test_prepare_history_does_not_compress_short_conversation():
    provider = FakeProvider()
    manager = ContextManager(provider)

    recent, summary, snapshot, compressed = manager.prepare_history(messages(4))

    assert len(recent) == 4
    assert summary == ""
    assert not compressed
    assert not snapshot.compressed
    assert provider.summary_calls == 0


def test_prepare_history_compresses_old_turns_and_keeps_recent_turns():
    provider = FakeProvider(token_scale=1)
    manager = ContextManager(provider)
    manager.SOFT_LIMIT_TOKENS = 10_000
    transcript = messages(10, width=4_000)

    recent, summary, snapshot, compressed = manager.prepare_history(transcript)

    assert compressed
    assert snapshot.compressed
    assert recent[-1] == transcript[-1]
    assert recent == transcript[-len(recent) :]
    assert summary.startswith("保留用户目标")
    assert provider.summary_calls == 1
    assert manager.last_report is not None
    assert manager.last_report.anchor_coverage == 1.0
    assert manager.last_report.verified


def test_prepare_history_only_merges_messages_not_already_in_summary():
    provider = FakeProvider(token_scale=1)
    manager = ContextManager(provider)
    manager.SOFT_LIMIT_TOKENS = 10_000
    transcript = messages(10, width=4_000)

    _, summary, _, compressed = manager.prepare_history(transcript)
    assert compressed
    assert provider.summary_calls == 1

    recent, same_summary, _, compressed_again = manager.prepare_history(
        transcript,
        summary=summary,
    )
    assert not compressed_again
    assert same_summary == summary
    assert provider.summary_calls == 1
    assert recent == transcript[-len(recent) :]

    extended = transcript + [
        {"role": "assistant", "content": "new assistant turn"},
        {"role": "user", "content": "new user turn"},
    ]
    manager.prepare_history(extended, summary=summary)

    assert provider.summary_calls == 2
    delta_prompt = provider.summary_prompts[-1].split("需要合并的新增历史：", 1)[1]
    assert "turn-4" in delta_prompt
    assert "turn-5" in delta_prompt
    assert "turn-0" not in delta_prompt
    assert manager.last_report.source_messages == 2


def test_compress_falls_back_locally_when_summary_model_fails():
    provider = FakeProvider(token_scale=1)
    provider.fail_summary = True
    manager = ContextManager(provider)

    summary = manager.compress(messages(8, width=100))

    assert "本地压缩" in summary
    assert "turn-0" in summary
    assert provider.summary_calls == 1
