from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from src.agents.coordinator import CoordinatorAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.extractor_agent import ExtractorAgent
from src.agents.state import AgentState
from src.core.config import AppConfig
from src.core.exceptions import IncompleteGenerationError
from src.extractors.inspector import DocumentInspector
from src.extractors.wechat import WeChatExtractor
from src.extractors.word import WordExtractor
from src.memory.conversation_store import ConversationStore


def test_word_extractor_reads_docx_with_installed_converter(tmp_path):
    source = tmp_path / "sample.docx"
    with ZipFile(source, "w") as document:
        document.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>",
        )
        document.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>",
        )
        document.writestr(
            "word/document.xml",
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>EchoStyle document check</w:t></w:r></w:p></w:body></w:document>",
        )

    result = WordExtractor().extract(str(source))
    assert "EchoStyle document check" in result


def test_legacy_doc_is_rejected_before_text_decoding(tmp_path):
    source = tmp_path / "legacy.doc"
    source.write_bytes(b"\xd0\xcf\x11\xe0")

    with pytest.raises(ValueError, match="转换为 .docx"):
        WordExtractor().extract(str(source))
    with pytest.raises(ValueError, match="转换为 .docx"):
        DocumentInspector.inspect(str(source))
    with pytest.raises(ValueError, match="转换为 .docx"):
        ExtractorAgent().run(AgentState(), str(source))


def test_wechat_challenge_is_not_treated_as_article():
    challenge = "Warning: This page maybe requiring CAPTCHA, please verify." + " filler" * 20
    with patch("src.extractors.wechat.httpx.Client") as client, patch(
        "src.extractors.wechat.trafilatura.fetch_url",
        return_value='<script src="https://captcha.gtimg.com/TCaptcha.js"></script>',
    ), patch("src.extractors.wechat.trafilatura.extract") as extract:
        client.return_value.__enter__.return_value.get.return_value = SimpleNamespace(
            status_code=200, text=challenge
        )
        with pytest.raises(RuntimeError, match="要求验证"):
            WeChatExtractor().extract("https://mp.weixin.qq.com/s/example")

    extract.assert_not_called()


def test_wechat_short_valid_fallback_remains_supported():
    challenge = "Warning: This page maybe requiring CAPTCHA, please verify." + " filler" * 20
    with patch("src.extractors.wechat.httpx.Client") as client, patch(
        "src.extractors.wechat.trafilatura.fetch_url", return_value="<html><article>valid</article></html>"
    ), patch("src.extractors.wechat.trafilatura.extract", return_value="一篇很短的文章。"):
        client.return_value.__enter__.return_value.get.return_value = SimpleNamespace(
            status_code=200, text=challenge
        )
        agent = ExtractorAgent()
        result = agent.run(AgentState(), "https://mp.weixin.qq.com/s/example")

    assert result["content"] == "一篇很短的文章。"
    assert result["engine_used"] == "trafilatura"


def test_wechat_jina_success_is_reported_as_jina():
    article = "# 正文\n\n" + "这是一段正常文章内容。" * 20
    with patch("src.extractors.wechat.httpx.Client") as client, patch(
        "src.extractors.wechat.trafilatura.fetch_url"
    ) as fallback:
        client.return_value.__enter__.return_value.get.return_value = SimpleNamespace(
            status_code=200, text=article
        )
        agent = ExtractorAgent()
        result = agent.run(AgentState(), "https://mp.weixin.qq.com/s/example")

    assert result["engine_used"] == "jina_reader"
    fallback.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_web_upload_removes_temporary_file(tmp_path, monkeypatch, fails):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    monkeypatch.chdir(tmp_path)
    payload = b"Sample article body"

    class StreamingUpload(BytesIO):
        name = "sample.md"

        def getvalue(self):
            raise AssertionError("streaming uploads must not call getvalue()")

    uploaded = StreamingUpload(payload)
    config = AppConfig()
    app_path = Path(__file__).resolve().parents[1] / "src/web/app.py"

    def extract(sources, state):
        assert len(sources) == 1
        assert Path(sources[0]).read_bytes() == payload
        if fails:
            raise RuntimeError("offline extraction failed")
        return [{"title": Path(sources[0]).stem, "content": "Sample article body", "engine_used": "plain_text", "char_count": 19}]

    chat_submission = SimpleNamespace(text="", files=[uploaded])
    with patch("src.core.config.load_config", return_value=config), patch.object(
        st, "chat_input", return_value=chat_submission
    ), patch.object(CoordinatorAgent, "extract_sources", side_effect=extract) as extractor:
        app = AppTest.from_file(app_path, default_timeout=10).run()
        assert not app.exception

    assert not app.exception
    assert extractor.call_args is not None
    assert not Path(extractor.call_args.args[0][0]).exists()
    assert len(app.session_state["documents"]) == (0 if fails else 1)
    if not fails:
        assert app.session_state["documents"][0]["title"] == "sample"
        persisted = ConversationStore(str(tmp_path / "profiles/conversation_state_v1.json")).load()
        assert persisted is not None
        assert persisted["documents"][0]["content"] == "Sample article body"
        assert persisted["messages"][0]["attachments"] == ["sample.md"]


def test_web_preserves_incomplete_draft_without_a_quality_report(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import streamlit as st

    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "src/web/app.py"
    with patch("src.core.config.load_config", return_value=AppConfig()), patch.object(
        st, "chat_input", return_value="写一篇一万字的文章"
    ), patch.object(
        ConversationAgent, "respond",
        side_effect=IncompleteGenerationError("保留下来的部分正文", 10_000, "续写达到上限"),
    ):
        app = AppTest.from_file(app_path, default_timeout=10).run()

    assert not app.exception
    message = app.session_state["messages"][-1]
    assert message["incomplete"] is True
    assert message["article"] == "保留下来的部分正文"
    assert message.get("report") is None
    assert app.session_state["last_article"] == "保留下来的部分正文"
    assert ConversationStore("profiles/conversation_state_v1.json").load()["messages"][-1] == message


def test_web_restores_persisted_conversation(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(tmp_path)
    state = ConversationStore.empty_state()
    state.update(
        {
            "messages": [
                {"role": "user", "content": "保留这个问题", "attachments": ["paper.md"]},
                {"role": "assistant", "content": "保留这个回答"},
            ],
            "documents": [
                {
                    "title": "paper",
                    "content": "重启后仍可引用的正文",
                    "document_id": "saved-hash",
                    "engine_used": "plain_text",
                    "char_count": 11,
                }
            ],
            "document_hashes": ["saved-hash"],
            "last_article": "保留的最后一稿",
            "context_summary": "保留的自动摘要",
            "summary_covered_messages": 1,
            "context_compression_count": 1,
            "context_notice": "已自动压缩较早的对话",
            "context_report": {"verified": True, "anchor_coverage": 1.0},
        }
    )
    ConversationStore("profiles/conversation_state_v1.json").save(state)
    app_path = Path(__file__).resolve().parents[1] / "src/web/app.py"

    with patch("src.core.config.load_config", return_value=AppConfig()):
        app = AppTest.from_file(app_path, default_timeout=10).run()

    assert not app.exception
    assert app.session_state["messages"] == state["messages"]
    assert app.session_state["documents"] == state["documents"]
    assert app.session_state["document_hashes"] == {"saved-hash"}
    assert app.session_state["context_summary"] == "保留的自动摘要"
    assert app.session_state["context_manager"].summary_covered_messages == 1


def test_web_migrates_legacy_context_manager_in_live_session(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.core.context_manager as context_manager_module

    class LegacyContextManager:
        _summary_covered_messages = 1
        model_provider = None

    monkeypatch.chdir(tmp_path)
    app_path = Path(__file__).resolve().parents[1] / "src/web/app.py"
    app = AppTest.from_file(app_path, default_timeout=10)
    app.session_state["context_manager"] = LegacyContextManager()
    app.session_state["messages"] = [
        {"role": "user", "content": "此前的写作要求"},
        {"role": "assistant", "content": "此前的回复"},
    ]
    app.session_state["context_summary"] = "此前的写作要求需要保留"

    with patch("src.core.config.load_config", return_value=AppConfig()), patch.object(
        context_manager_module, "ContextManager", LegacyContextManager
    ):
        app.run()

    assert not app.exception
    assert app.session_state["messages"][0]["content"] == "此前的写作要求"
    assert app.session_state["context_manager"].summary_covered_messages == 1
    assert ConversationStore("profiles/conversation_state_v1.json").load()["context_summary"] == "此前的写作要求需要保留"
