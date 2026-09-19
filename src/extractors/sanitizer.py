import html
import re
from typing import List


class TextSanitizer:
    """
    文本去噪与清洗管道：
    专门针对文风学习过滤无意义的平台引导语、修补分行断裂，并保留核心标点与分段呼吸感。
    """

    # 微信/自媒体常见平台噪音正则列表
    NOISE_PATTERNS: List[re.Pattern] = [
        re.compile(r"点击上方[“\"\'\s]?.*?关注.*?", re.IGNORECASE),
        re.compile(r"长按(?:扫描|识别)?二维码.*?关注.*?", re.IGNORECASE),
        re.compile(r"喜欢(?:本文|这篇文章)?.*?点个?(?:在看|赞|分享|收藏).*?", re.IGNORECASE),
        re.compile(r"作者\s*[:：].*?(?:\n|$)", re.IGNORECASE),
        re.compile(r"本文(?:系|为)?(?:原创|独家).*?(?:禁止转载|授权).*?(?:\n|$)", re.IGNORECASE),
        re.compile(r"往期(?:精彩)?推荐[\s\S]*?$", re.IGNORECASE),  # 往往在文章最末尾
        re.compile(r"免责声明[\s\S]*?$", re.IGNORECASE),
        re.compile(r"阅读原文", re.IGNORECASE),
        re.compile(r"<!--[\s\S]*?-->"),  # 移除 HTML 注释
    ]

    @classmethod
    def clean(
        cls,
        text: str,
        repair_linebreaks: bool = True,
        clean_pdf_artifacts: bool = False,
    ) -> str:
        """
        执行完整清洗流水线
        """
        if not text:
            return ""

        if clean_pdf_artifacts:
            text = cls._clean_pdf_artifacts(text)

        # 1. 移除 Markdown 图片标记 (如 ![图片说明](url)) 以及文字 [图片] 占位符
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = re.sub(r"\[(图片|插图|图|视频|音频)\]", "", text)

        # 2. 逐条过滤平台噪音
        for pattern in cls.NOISE_PATTERNS:
            text = pattern.sub("", text)

        # 3. 修复 PDF 造成的断行（如果开启）
        if repair_linebreaks:
            text = cls._repair_broken_lines(text)

        # 4. 压缩过多的连续空行与多余空格
        text = re.sub(r"[ \t]+", " ", text)  # 行内多余空格
        text = re.sub(r"\n{3,}", "\n\n", text)  # 超过 2 个换行压缩为 2 个

        return text.strip()

    @staticmethod
    def _clean_pdf_artifacts(text: str) -> str:
        """清除 PDF 转 Markdown 常见的控制字符、cid 占位符和 HTML 包装。"""
        # MinerU 偶尔会把字体编码残留写成不可见控制字符，避免污染统计特征。
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200B-\u200F\u202A-\u202E]", "", text)
        text = re.sub(r"\(cid:\s*\d+\)", "", text, flags=re.IGNORECASE)
        text = html.unescape(text).replace("\u00a0", " ")

        # 保留脚注/上标中的文字，只移除 MinerU 添加的 HTML 标签本身。
        text = re.sub(r"<small[^>]*>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</small>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</?(?:span|sup|sub)(?:\s[^>]*)?>", "", text, flags=re.IGNORECASE)
        return text

    @classmethod
    def _repair_broken_lines(cls, text: str) -> str:
        """
        修补因 PDF 视觉换行导致的非自然断句：
        - 英文连字符换行：例如 "connec-\ntion" -> "connection"
        - 中文跨行断句：若上一行结尾不是 。！？!?…：“” 等句末/开闭标点，且下一行开头也是正文字符，则合并为一行。
        """
        # 处理英文连字符换行
        text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

        # 处理中文断行 (避开 Markdown 列表与标题符号)
        lines = text.split("\n")
        repaired_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            # 如果当前行是空的，直接保留
            if not line.strip():
                repaired_lines.append(line)
                i += 1
                continue

            # 如果当前行是 Markdown 标题(#)或引用(>)或列表(-, *, 1.)，不主动向上合并
            if re.match(r"^\s*(#{1,6}|>|[-*+]|\d+\.)\s+", line):
                repaired_lines.append(line)
                i += 1
                continue

            # 检查下一行是否需要与当前行合并
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                # 若下一行为空，说明是明确的段落分段，保留不合并
                if not next_line.strip():
                    break
                # 若下一行是 Markdown 结构（标题、引用、列表），不合并
                if re.match(r"^\s*(#{1,6}|>|[-*+]|\d+\.)\s+", next_line):
                    break

                # 检查当前行末字符
                last_char = line.rstrip()[-1] if line.rstrip() else ""
                # 中文完整标点
                terminators = ("。", "！", "？", "!", "?", "…", "；", ";", "：", ":", "\"", "”", "’")

                if last_char and (last_char not in terminators):
                    # 说明句子被生硬折断了，进行平滑合并
                    line = line.rstrip() + next_line.lstrip()
                    i += 1
                else:
                    break

            repaired_lines.append(line)
            i += 1

        return "\n".join(repaired_lines)
