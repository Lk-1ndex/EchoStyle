import os
from pathlib import Path
from markitdown import MarkItDown
from .base import BaseExtractor
from .sanitizer import TextSanitizer


class WordExtractor(BaseExtractor):
    """
    Word 文档 (.docx) 提取器：
    利用微软 MarkItDown 引擎将 Word 格式无损转换为结构化 Markdown。
    """

    def __init__(self):
        self.md_engine = MarkItDown()

    def extract(self, file_path: str) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        if path.suffix.lower() not in [".docx", ".doc"]:
            raise ValueError(f"不支持的文件扩展名: {path.suffix}，仅支持 .docx 或 .doc")

        # 使用 MarkItDown 统一转译
        conversion_result = self.md_engine.convert(str(path))
        raw_text = conversion_result.text_content

        # 进行文本清洗
        cleaned_text = TextSanitizer.clean(raw_text, repair_linebreaks=False)
        return cleaned_text
