from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

import pytest

from src.agents.coordinator import CoordinatorAgent
from src.agents.extractor_agent import ExtractorAgent
from src.agents.state import AgentState
from src.core.config import AppConfig
from src.extractors.inspector import DocumentInspector
from src.extractors.wechat import WeChatExtractor
from src.extractors.word import WordExtractor


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
    uploaded = SimpleNamespace(name="sample.md", getvalue=lambda: b"Sample article body")
    config = AppConfig()
    app_path = Path(__file__).resolve().parents[1] / "src/web/app.py"

    def extract(sources, state):
        assert len(sources) == 1
        assert Path(sources[0]).read_bytes() == uploaded.getvalue()
        if fails:
            raise RuntimeError("offline extraction failed")
        return [{"title": Path(sources[0]).stem, "content": "Sample article body", "engine_used": "plain_text", "char_count": 19}]

    with patch("src.core.config.load_config", return_value=config), patch.object(
        st, "file_uploader", return_value=[uploaded]
    ), patch.object(CoordinatorAgent, "extract_sources", side_effect=extract) as extractor:
        app = AppTest.from_file(app_path, default_timeout=10).run()
        assert not app.exception
        next(button for button in app.button if "Extractor Agent" in button.label).click().run()

    assert not app.exception
    assert extractor.call_args is not None
    assert not Path(extractor.call_args.args[0][0]).exists()
    assert len(app.session_state["samples"]) == (0 if fails else 1)
    if not fails:
        assert app.session_state["samples"][0]["title"] == "sample"
