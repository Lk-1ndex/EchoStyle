from pathlib import Path
from subprocess import CalledProcessError, TimeoutExpired
from unittest.mock import Mock, patch

from src.agents.extractor_agent import ExtractorAgent
from src.agents.state import AgentState
from src.core.config import ExtractorConfig
from src.extractors.pdf import PDFExtractor


def test_mineru_4_cli_reads_generated_markdown(tmp_path):
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    extractor = PDFExtractor(engine="mineru", mineru_tier="standard")

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1]) / "sample" / "sample.md"
        output.parent.mkdir(parents=True)
        output.write_text("# MinerU result", encoding="utf-8")
        return Mock(returncode=0)

    with patch("src.extractors.pdf.shutil.which", return_value="mineru-kit"), patch(
        "src.extractors.pdf.subprocess.run", side_effect=fake_run
    ) as run:
        result = extractor.extract(str(source))

    command = run.call_args.args[0]
    assert command[:3] == ["mineru-kit", "parse", str(source)]
    assert command[command.index("--tier") + 1] == "standard"
    assert run.call_args.kwargs["encoding"] == "utf-8"
    assert run.call_args.kwargs["errors"] == "replace"
    assert result == "# MinerU result"
    assert extractor.last_engine_used == "mineru"


def test_pdf_defaults_and_legacy_auto_alias_to_mineru():
    assert PDFExtractor().engine == "mineru"
    assert PDFExtractor(engine="auto").engine == "mineru"
    assert ExtractorConfig().pdf_engine == "mineru"


def test_mineru_resource_limits_are_passed_to_subprocess(tmp_path):
    source = tmp_path / "limits.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    extractor = PDFExtractor(
        engine="mineru",
        mineru_timeout=17,
        mineru_intra_op_num_threads=3,
        mineru_inter_op_num_threads=2,
        mineru_pdf_render_threads=4,
        mineru_malloc_trim=True,
    )

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1]) / "limits.md"
        output.write_text("# output", encoding="utf-8")
        return Mock(returncode=0)

    with patch("src.extractors.pdf.shutil.which", return_value="mineru-kit"), patch(
        "src.extractors.pdf.subprocess.run", side_effect=fake_run
    ) as run:
        assert extractor.extract(str(source)) == "# output"

    kwargs = run.call_args.kwargs
    assert kwargs["timeout"] == 17
    assert kwargs["env"]["MINERU_INTRA_OP_NUM_THREADS"] == "3"
    assert kwargs["env"]["MINERU_INTER_OP_NUM_THREADS"] == "2"
    assert kwargs["env"]["MINERU_PDF_RENDER_THREADS"] == "4"
    assert kwargs["env"]["MINERU_MALLOC_TRIM"] == "1"


def test_legacy_magic_pdf_config_uses_mineru_4(tmp_path):
    source = tmp_path / "legacy.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    extractor = PDFExtractor(engine="mineru", mineru_cmd="magic-pdf")

    def which(command):
        return "mineru-kit" if command == "mineru-kit" else None

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--output") + 1]) / "result.md"
        output.write_text("legacy config works", encoding="utf-8")
        return Mock(returncode=0)

    with patch("src.extractors.pdf.shutil.which", side_effect=which), patch(
        "src.extractors.pdf.subprocess.run", side_effect=fake_run
    ) as run:
        assert extractor.extract(str(source)) == "legacy config works"

    assert run.call_args.args[0][0:2] == ["mineru-kit", "parse"]


def test_extractor_agent_auto_never_downgrades_by_layout_recommendation(tmp_path):
    source = tmp_path / "single_column_scientific.pdf"
    source.write_bytes(b"%PDF-1.4\n")

    with patch("src.agents.extractor_agent.DocumentInspector.inspect", return_value={
        "recommended_engine": "markitdown",
        "reason": "标准单栏，但仅作诊断",
    }), patch("src.agents.extractor_agent.PDFExtractor") as pdf_cls:
        pdf_cls.return_value.extract.return_value = "# MinerU output"
        pdf_cls.return_value.last_engine_used = "mineru"
        result = ExtractorAgent().run(AgentState(), str(source), force_engine="auto")

    assert result["engine_used"] == "pdf_mineru"
    assert pdf_cls.call_args.kwargs["engine"] == "mineru"


def test_extractor_agent_reports_actual_fallback_engine(tmp_path):
    source = tmp_path / "fallback.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    agent = ExtractorAgent(mineru_cmd="missing-mineru")

    with patch("src.agents.extractor_agent.DocumentInspector.inspect", return_value={
        "recommended_engine": "mineru",
        "reason": "test",
    }), patch("src.extractors.pdf.shutil.which", return_value=None), patch.object(
        PDFExtractor, "_extract_with_markitdown", return_value="fallback text"
    ):
        result = agent.run(AgentState(), str(source), force_engine="auto")

    assert result["engine_used"] == "pdf_markitdown"
    assert "fallback_reason" in result


def test_mineru_failure_reports_reason_and_uses_markitdown(tmp_path, capsys):
    source = tmp_path / "failure.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    extractor = PDFExtractor(engine="mineru")
    failure = CalledProcessError(1, ["mineru-kit"], stderr="model load failed: bad allocation")

    with patch("src.extractors.pdf.shutil.which", return_value="mineru-kit"), patch(
        "src.extractors.pdf.subprocess.run", side_effect=failure
    ), patch.object(PDFExtractor, "_extract_with_markitdown", return_value="fallback text"):
        assert extractor.extract(str(source)) == "fallback text"

    assert "bad allocation" in capsys.readouterr().out
    assert extractor.last_engine_used == "markitdown"
    assert "bad allocation" in extractor.last_fallback_reason


def test_mineru_timeout_falls_back_to_markitdown(tmp_path):
    source = tmp_path / "timeout.pdf"
    source.write_bytes(b"%PDF-1.4\n")
    extractor = PDFExtractor(engine="mineru", mineru_timeout=1)
    timeout = TimeoutExpired(["mineru-kit"], 1)

    with patch("src.extractors.pdf.shutil.which", return_value="mineru-kit"), patch(
        "src.extractors.pdf.subprocess.run", side_effect=timeout
    ), patch.object(PDFExtractor, "_extract_with_markitdown", return_value="fallback text"):
        assert extractor.extract(str(source)) == "fallback text"

    assert extractor.last_engine_used == "markitdown"
    assert "超时" in extractor.last_fallback_reason
