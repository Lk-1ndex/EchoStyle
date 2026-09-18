import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from markitdown import MarkItDown
from .base import BaseExtractor
from .sanitizer import TextSanitizer


class PDFExtractor(BaseExtractor):
    """
    PDF 文档提取器：
    双引擎架构：
    1. 'markitdown' 模式 (默认)：极速、轻量，适用于大多数标准单栏电子版 PDF。
    2. 'mineru' 模式：调用 MinerU 4 (mineru-kit) 深度视觉解析，适用于复杂双栏排版、学术论文及扫描件。
    """

    def __init__(
        self,
        engine: str = "markitdown",
        mineru_cmd: str = "mineru-kit",
        mineru_tier: str = "basic",
    ):
        self.engine = engine.lower()
        self.mineru_cmd = mineru_cmd
        self.mineru_tier = mineru_tier.lower()
        self.last_engine_used = "markitdown"
        self.md_engine = MarkItDown()

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"不是合法的 PDF 文件: {file_path}")

        raw_md: str = ""

        # 模式 1：MinerU 深度视觉引擎
        if self.engine == "mineru":
            raw_md = self._extract_with_mineru(path)
        else:
            # 模式 2：默认 MarkItDown 极速引擎
            raw_md = self._extract_with_markitdown(path)

        # 经过管道修补断行与清洗
        cleaned_md = TextSanitizer.clean(raw_md, repair_linebreaks=True)
        return cleaned_md

    def _extract_with_markitdown(self, path: Path) -> str:
        try:
            res = self.md_engine.convert(str(path))
            self.last_engine_used = "markitdown"
            return res.text_content
        except Exception as e:
            raise RuntimeError(f"MarkItDown 解析 PDF 失败: {str(e)}")

    def _extract_with_mineru(self, path: Path) -> str:
        """调用本地 MinerU CLI；兼容旧配置中的 magic-pdf 命令名。"""
        command = self.mineru_cmd
        if not shutil.which(command) and command == "magic-pdf" and shutil.which("mineru-kit"):
            command = "mineru-kit"

        if not shutil.which(command):
            print(f"[提示] 未找到 {command} 命令行，自动回退到 MarkItDown 引擎解析。")
            return self._extract_with_markitdown(path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            if Path(command).name.lower() in {"mineru-kit", "mineru-kit.exe"}:
                cmd = [
                    command,
                    "parse",
                    str(path),
                    "--output",
                    tmp_dir,
                    "--format",
                    "markdown",
                    "--tier",
                    self.mineru_tier,
                ]
            else:
                cmd = [command, "-p", str(path), "-o", tmp_dir, "-m", "auto"]
            try:
                subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
                candidates = sorted(Path(tmp_dir).rglob("*.md"))
                if candidates:
                    self.last_engine_used = "mineru"
                    return candidates[0].read_text(encoding="utf-8")

                print("[警告] MinerU 未生成 Markdown，回退至 MarkItDown。")
                return self._extract_with_markitdown(path)
            except Exception as e:
                detail = (getattr(e, "stderr", None) or str(e)).strip()
                print(f"[警告] MinerU 解析出错 ({detail[-500:]})，回退至 MarkItDown。")
                return self._extract_with_markitdown(path)
