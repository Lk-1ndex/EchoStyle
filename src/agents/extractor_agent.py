from pathlib import Path
from typing import Any, Dict
from .base import BaseAgent
from .state import AgentState, AgentStatus
from src.extractors.inspector import DocumentInspector
from src.extractors.wechat import WeChatExtractor
from src.extractors.word import WordExtractor
from src.extractors.pdf import PDFExtractor
from src.extractors.sanitizer import TextSanitizer


class ExtractorAgent(BaseAgent):
    """
    文档感知与高精提取智能体：
    负责来源识别、PDF MinerU 解析和统一文本清洗，并驱动状态机。
    """

    def __init__(
        self,
        mineru_cmd: str = "mineru-kit",
        mineru_tier: str = "basic",
        mineru_timeout: int = 900,
        mineru_intra_op_num_threads: int = 2,
        mineru_inter_op_num_threads: int = 1,
        mineru_pdf_render_threads: int = 1,
        mineru_malloc_trim: bool = True,
    ):
        super().__init__(name="ExtractorAgent", description="负责来源识别、MinerU 解析与去噪抽取")
        self.mineru_cmd = mineru_cmd
        self.mineru_tier = mineru_tier
        self.mineru_timeout = mineru_timeout
        self.mineru_intra_op_num_threads = mineru_intra_op_num_threads
        self.mineru_inter_op_num_threads = mineru_inter_op_num_threads
        self.mineru_pdf_render_threads = mineru_pdf_render_threads
        self.mineru_malloc_trim = mineru_malloc_trim
        self.wechat_extractor = WeChatExtractor()
        self.word_extractor = WordExtractor()

    def run(self, state: AgentState, source: str, force_engine: str = "mineru") -> Dict[str, Any]:
        source = source.strip()
        state.transition_to(AgentStatus.PARSING, f"开始处理样文输入源: {source}")

        # 1. 微信公众号 URL
        if source.startswith("http://") or source.startswith("https://"):
            state.transition_to(AgentStatus.PARSING, "探测为微信公众号网络链接，调用 Jina 高保真转译引擎...")
            clean_md = self.wechat_extractor.extract(source)
            return {
                "title": f"微信文章_{source[-10:]}",
                "content": clean_md,
                "engine_used": self.wechat_extractor.last_engine_used,
                "char_count": len(clean_md),
            }

        p = Path(source)
        if not p.exists():
            state.record_error(f"文件未找到: {source}")
            raise FileNotFoundError(f"文件未找到: {source}")

        ext = p.suffix.lower()

        # 2. Word 文档
        if ext in [".docx", ".doc"]:
            state.transition_to(AgentStatus.PARSING, "探测为 Word 文档，使用 MarkItDown 进行语义结构解析...")
            clean_md = self.word_extractor.extract(str(p))
            return {
                "title": p.stem,
                "content": clean_md,
                "engine_used": "markitdown_word",
                "char_count": len(clean_md),
            }

        # 3. PDF 文档：统一使用 MinerU；auto 仅作为旧配置兼容别名。
        if ext == ".pdf":
            inspection = DocumentInspector.inspect(str(p))
            requested_engine = (force_engine or "mineru").strip().lower()
            chosen_engine = "mineru" if requested_engine == "auto" else requested_engine
            if chosen_engine == "mineru":
                route_message = "PDF 统一使用 MinerU（MarkItDown 仅作失败回退）"
            else:
                route_message = f"显式指定 PDF 对照引擎 [{chosen_engine}]"
            state.transition_to(
                AgentStatus.PARSING,
                f"PDF 版面诊断: {inspection['reason']} => {route_message}",
            )

            pdf_extractor = PDFExtractor(
                engine=chosen_engine,
                mineru_cmd=self.mineru_cmd,
                mineru_tier=self.mineru_tier,
                mineru_timeout=self.mineru_timeout,
                mineru_intra_op_num_threads=self.mineru_intra_op_num_threads,
                mineru_inter_op_num_threads=self.mineru_inter_op_num_threads,
                mineru_pdf_render_threads=self.mineru_pdf_render_threads,
                mineru_malloc_trim=self.mineru_malloc_trim,
            )
            clean_md = pdf_extractor.extract(str(p))
            result = {
                "title": p.stem,
                "content": clean_md,
                "engine_used": f"pdf_{pdf_extractor.last_engine_used}",
                "char_count": len(clean_md),
            }
            if pdf_extractor.last_fallback_reason:
                result["fallback_reason"] = pdf_extractor.last_fallback_reason
                state.execution_logs.append(
                    f"[PDF FALLBACK] MinerU 未完成解析，已回退 MarkItDown: {pdf_extractor.last_fallback_reason}"
                )
            return result

        # 4. 普通纯文本
        raw_text = p.read_text(encoding="utf-8")
        clean_md = TextSanitizer.clean(raw_text)
        return {
            "title": p.stem,
            "content": clean_md,
            "engine_used": "plain_text",
            "char_count": len(clean_md),
        }
