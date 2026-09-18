from pathlib import Path
from typing import Dict, Any


class DocumentInspector:
    """
    文档复杂度与排版感知器：
    自动分析输入文件的物理特征与排版复杂性，智能决策最佳解析引擎（MarkItDown 极速 vs MinerU 深度视觉）。
    """

    @classmethod
    def inspect(cls, file_path: str) -> Dict[str, Any]:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = p.suffix.lower()
        result = {
            "file_name": p.name,
            "extension": ext,
            "is_scanned": False,
            "is_complex_layout": False,
            "recommended_engine": "markitdown",
            "reason": "常规单栏文档",
        }

        if ext == ".doc":
            raise ValueError("旧版 .doc 文件暂不支持，请先转换为 .docx。")

        if ext in [".docx", ".txt", ".md"]:
            result["recommended_engine"] = "markitdown"
            result["reason"] = "Word/纯文本结构清晰，极速引擎即可实现 100% 保真"
            return result

        if ext == ".pdf":
            # 针对 PDF 进行排版探测
            cls._inspect_pdf(p, result)

        return result

    @classmethod
    def _inspect_pdf(cls, path: Path, result: Dict[str, Any]):
        try:
            import pymupdf
            doc = pymupdf.open(str(path))
            total_pages = len(doc)
            sample_pages = doc[:min(5, total_pages)]

            total_chars = 0
            has_multi_column = False

            for page in sample_pages:
                text = page.get_text("text")
                total_chars += len(text.strip())

                # 简单双栏探测：分析文本块的水平分布
                blocks = page.get_text("blocks")
                # 如果同一页内有多个并列在左右不同 X 轴区间的文本块
                x_positions = [b[0] for b in blocks if len(b) > 4 and len(str(b[4]).strip()) > 20]
                if len(x_positions) >= 4:
                    left_count = sum(1 for x in x_positions if x < page.rect.width * 0.45)
                    right_count = sum(1 for x in x_positions if x > page.rect.width * 0.50)
                    if left_count >= 2 and right_count >= 2:
                        has_multi_column = True

            doc.close()

            # 判断是否为扫描件
            avg_chars_per_page = total_chars / max(1, len(sample_pages))
            if avg_chars_per_page < 30:
                result["is_scanned"] = True
                result["is_complex_layout"] = True
                result["recommended_engine"] = "mineru"
                result["reason"] = "检测到无文本层或字符极少，判定为纯图片扫描件，需使用 MinerU 深度视觉 OCR 解析"
            elif has_multi_column:
                result["is_complex_layout"] = True
                result["recommended_engine"] = "mineru"
                result["reason"] = "检测到疑似双栏/学术排版，推荐 MinerU 深度视觉引擎重构人类阅读流"
            else:
                result["recommended_engine"] = "markitdown"
                result["reason"] = "检测到标准单栏电子版 PDF，字符流完整，使用极速引擎秒级解析"

        except Exception as e:
            # 容错降级
            result["recommended_engine"] = "markitdown"
            result["reason"] = f"排版探测跳过 ({str(e)})，默认使用 MarkItDown"
