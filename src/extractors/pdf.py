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
    PDF 文档提取器。

    PDF 生产路径统一优先使用 MinerU 4 (mineru-kit) 深度视觉解析；
    MarkItDown 只在 MinerU 不可用、超时或没有生成有效 Markdown 时作为故障回退。
    """

    DEFAULT_MINERU_TIMEOUT = 900

    def __init__(
        self,
        engine: str = "mineru",
        mineru_cmd: str = "mineru-kit",
        mineru_tier: str = "basic",
        mineru_timeout: int = DEFAULT_MINERU_TIMEOUT,
        mineru_intra_op_num_threads: int = 2,
        mineru_inter_op_num_threads: int = 1,
        mineru_pdf_render_threads: int = 1,
        mineru_malloc_trim: bool = True,
    ):
        normalized_engine = (engine or "mineru").strip().lower()
        # 兼容旧版 config.yaml，但不再根据版面自动切换到 MarkItDown。
        if normalized_engine == "auto":
            normalized_engine = "mineru"
        if normalized_engine not in {"mineru", "markitdown"}:
            raise ValueError(f"不支持的 PDF 解析引擎: {engine!r}，可选 mineru 或 markitdown")

        self.engine = normalized_engine
        self.mineru_cmd = mineru_cmd
        self.mineru_tier = mineru_tier.lower()
        self.mineru_timeout = max(1, int(mineru_timeout))
        self.mineru_intra_op_num_threads = max(1, int(mineru_intra_op_num_threads))
        self.mineru_inter_op_num_threads = max(1, int(mineru_inter_op_num_threads))
        self.mineru_pdf_render_threads = max(1, int(mineru_pdf_render_threads))
        self.mineru_malloc_trim = bool(mineru_malloc_trim)
        self.last_engine_used = "markitdown"
        self.last_fallback_reason = ""
        self.md_engine = MarkItDown()

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"不是合法的 PDF 文件: {file_path}")

        self.last_fallback_reason = ""
        if self.engine == "mineru":
            raw_md = self._extract_with_mineru(path)
        else:
            # 仅供显式诊断/对照使用；MinerU 失败回退也会经过这里。
            raw_md = self._extract_with_markitdown(path)

        # MinerU/MarkItDown 都可能带出 PDF 控制字符、HTML 包装和 cid 残留。
        return TextSanitizer.clean(
            raw_md,
            repair_linebreaks=True,
            clean_pdf_artifacts=True,
        )

    def _extract_with_markitdown(self, path: Path) -> str:
        try:
            res = self.md_engine.convert(str(path))
            self.last_engine_used = "markitdown"
            return res.text_content
        except Exception as e:
            raise RuntimeError(f"MarkItDown 解析 PDF 失败: {str(e)}") from e

    def _extract_with_mineru(self, path: Path) -> str:
        """调用本地 MinerU CLI；兼容旧配置中的 magic-pdf 命令名。"""
        command = self.mineru_cmd
        if not shutil.which(command) and command == "magic-pdf" and shutil.which("mineru-kit"):
            command = "mineru-kit"

        if not shutil.which(command):
            return self._fallback_to_markitdown(
                path,
                f"未找到 MinerU 命令行: {command}",
            )

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
                    timeout=self.mineru_timeout,
                    env=self._mineru_environment(),
                )
                candidates = sorted(Path(tmp_dir).rglob("*.md"))
                if candidates:
                    content = candidates[0].read_text(encoding="utf-8", errors="replace")
                    if content.strip():
                        self.last_engine_used = "mineru"
                        self.last_fallback_reason = ""
                        return content

                return self._fallback_to_markitdown(
                    path,
                    "MinerU 未生成非空 Markdown",
                )
            except subprocess.TimeoutExpired as e:
                return self._fallback_to_markitdown(
                    path,
                    f"MinerU 超时（>{self.mineru_timeout}s）: {self._process_detail(e)}",
                )
            except Exception as e:
                return self._fallback_to_markitdown(
                    path,
                    f"MinerU 解析出错: {self._process_detail(e)}",
                )

    def _mineru_environment(self) -> dict:
        """为 MinerU 设置保守的 CPU/渲染线程上限，降低峰值内存。"""
        env = os.environ.copy()
        env.update(
            {
                "MINERU_INTRA_OP_NUM_THREADS": str(self.mineru_intra_op_num_threads),
                "MINERU_INTER_OP_NUM_THREADS": str(self.mineru_inter_op_num_threads),
                "MINERU_PDF_RENDER_THREADS": str(self.mineru_pdf_render_threads),
            }
        )
        if self.mineru_malloc_trim:
            env["MINERU_MALLOC_TRIM"] = "1"
        else:
            env.pop("MINERU_MALLOC_TRIM", None)
        return env

    @staticmethod
    def _process_detail(error: BaseException) -> str:
        detail = getattr(error, "stderr", None) or getattr(error, "output", None) or str(error)
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        return str(detail).strip()[-500:]

    def _fallback_to_markitdown(self, path: Path, reason: str) -> str:
        """记录 MinerU 故障原因后执行最后的轻量回退。"""
        self.last_fallback_reason = reason
        print(f"[警告] {reason}，回退至 MarkItDown。")
        try:
            return self._extract_with_markitdown(path)
        except Exception as fallback_error:
            raise RuntimeError(
                f"MinerU 失败且 MarkItDown 回退失败: {reason}; {fallback_error}"
            ) from fallback_error
