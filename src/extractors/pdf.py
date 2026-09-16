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
    2. 'mineru' 模式：调用 opendatalab/MinerU (magic-pdf) 深度视觉解析，适用于复杂双栏排版、学术论文及扫描件。
    """

    def __init__(self, engine: str = "markitdown", mineru_cmd: str = "magic-pdf"):
        self.engine = engine.lower()
        self.mineru_cmd = mineru_cmd
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
            return res.text_content
        except Exception as e:
            raise RuntimeError(f"MarkItDown 解析 PDF 失败: {str(e)}")

    def _extract_with_mineru(self, path: Path) -> str:
        """调用本地安装的 magic-pdf 命令行工具"""
        if not shutil.which(self.mineru_cmd):
            # 如果本地未安装 magic-pdf 命令，自动回退到 markitdown 并给出提示
            print(f"[提示] 未找到 {self.mineru_cmd} 命令行，自动回退到 MarkItDown 引擎解析。")
            return self._extract_with_markitdown(path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            cmd = [
                self.mineru_cmd,
                "-p", str(path),
                "-o", tmp_dir,
                "-m", "auto"
            ]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                # 寻找输出目录下的 auto.md
                stem = path.stem
                output_folder = Path(tmp_dir) / stem / "auto"
                output_md = output_folder / f"{stem}.md"
                if not output_md.exists():
                    # 备选路径检查
                    for candidate in Path(tmp_dir).rglob("*.md"):
                        output_md = candidate
                        break

                if output_md.exists():
                    return output_md.read_text(encoding="utf-8")
                else:
                    return self._extract_with_markitdown(path)
            except Exception as e:
                print(f"[警告] MinerU 解析出错 ({str(e)})，回退至 MarkItDown。")
                return self._extract_with_markitdown(path)
