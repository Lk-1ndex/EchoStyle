from src.core.config import LLMConfig
from src.core.model_provider import ModelProvider


def test_deepseek_flash_defaults_and_context_window():
    config = LLMConfig()
    provider = ModelProvider(config)

    assert config.base_url == "https://api.deepseek.com"
    assert config.model == "deepseek-flash"
    assert ModelProvider.MODEL_CONTEXT_WINDOWS["deepseek-flash"] == 1_000_000
    assert provider.effective_context_window() == 1_000_000
    assert provider.input_token_budget() == 994_904


def test_explicit_context_window_overrides_unknown_model_default():
    provider = ModelProvider(
        LLMConfig(model="custom-provider-model", context_window=200_000, max_tokens=4_096)
    )

    assert provider.effective_context_window() == 200_000
    assert provider.input_token_budget() == 194_904


def test_dynamic_token_budget_modes():
    provider = ModelProvider(LLMConfig(api_key="mock_key", model="gpt-4o"))

    persona = "你是一位犀利毒舌的独立评论家。" * 50
    style_dna = "平均句长18字，长短句交替跌宕，多用设问与金句。" * 50
    memory = ["范例段落内容：" + ("很多人的文章毫无破绽却也毫无见解。" * 20)] * 10
    task = "写一篇关于技术反思的文章，重点抨击无脑搬运。" * 20
    feedback = "第一版句式偏长，缺少短句爆破点。" * 20

    # 1. 常规长文创作 write 模式
    sys_w, user_w = provider.assemble_budgeted_prompt(
        persona, style_dna, memory, task, feedback, task_mode="write"
    )
    assert "独立评论家" in sys_w
    assert "高光范例" in sys_w
    assert "反思修正" in user_w

    # 2. 改写 rewrite 模式：任务与草稿预算大幅增加，记忆减少
    sys_rw, user_rw = provider.assemble_budgeted_prompt(
        persona, style_dna, memory, task, feedback, task_mode="rewrite"
    )
    assert len(user_rw) > len(user_w) or "反思修正" in user_rw

    # 3. 社媒短帖 short_post 模式：记忆范例预算最高
    sys_sp, user_sp = provider.assemble_budgeted_prompt(
        persona, style_dna, memory, task, feedback, task_mode="short_post"
    )
    assert "高光范例" in sys_sp


def test_unused_section_budgets_are_reclaimed_for_the_current_task():
    provider = ModelProvider(LLMConfig(api_key="mock_key", model="gpt-4o"))
    long_task = "context " * 40_000

    system_prompt, user_prompt = provider.assemble_budgeted_prompt(
        system_persona="简洁写作助手",
        style_dna="",
        memory_exemplars=[],
        user_task=long_task,
        task_mode="write",
    )

    old_fixed_task_quota = int(provider.input_token_budget() * 0.15)
    assert provider.count_tokens(user_prompt) > old_fixed_task_quota
    assert provider.count_tokens(system_prompt) + provider.count_tokens(user_prompt) <= provider.input_token_budget()
