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
    具备自动感知文档排版复杂度的能力，自适应分派最佳解析引擎，并驱动状态机。
    """

    def __init__(self, mineru_cmd: str = "magic-pdf"):
        super().__init__(name="ExtractorAgent", description="负责文档排版探测、引擎自适应分派与去噪抽取")
        self.mineru_cmd = mineru_cmd
        self.wechat_extractor = WeChatExtractor()
        self.word_extractor = WordExtractor()

    def run(self, state: AgentState, source: str, force_engine: str = "auto") -> Dict[str, Any]:
        source = source.strip()
        state.transition_to(AgentStatus.PARSING, f"开始处理样文输入源: {source}")

        # 1. 微信公众号 URL
        if source.startswith("http://") or source.startswith("https://"):
            state.transition_to(AgentStatus.PARSING, "探测为微信公众号网络链接，调用 Jina 高保真转译引擎...")
            clean_md = self.wechat_extractor.extract(source)
            return {
                "title": f"微信文章_{source[-10:]}",
                "content": clean_md,
                "engine_used": "jina_reader",
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

        # 3. PDF 文档 (自适应复杂度路由)
        if ext == ".pdf":
            inspection = DocumentInspector.inspect(str(p))
            chosen_engine = inspection["recommended_engine"] if force_engine == "auto" else force_engine
            state.transition_to(AgentStatus.PARSING, f"PDF 排版探测: {inspection['reason']} => 决策选用引擎 [{chosen_engine}]")

            pdf_extractor = PDFExtractor(engine=chosen_engine, mineru_cmd=self.mineru_cmd)
            clean_md = pdf_extractor.extract(str(p))
            return {
                "title": p.stem,
                "content": clean_md,
                "engine_used": f"pdf_{chosen_engine}",
                "char_count": len(clean_md),
            }

        # 4. 普通纯文本
        raw_text = p.read_text(encoding="utf-8")
        clean_md = TextSanitizer.clean(raw_text)
        return {
            "title": p.stem,
            "content": clean_md,
            "engine_used": "plain_text",
            "char_count": len(clean_md),
        }
