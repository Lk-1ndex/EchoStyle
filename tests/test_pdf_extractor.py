from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import Mock, patch

from src.agents.extractor_agent import ExtractorAgent
from src.agents.state import AgentState
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
