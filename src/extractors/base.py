from abc import ABC, abstractmethod
from typing import Optional


class BaseExtractor(ABC):
    """文档提取器基类"""

    @abstractmethod
    def extract(self, source: str) -> str:
        """
        从输入源提取纯净的 Markdown 正文。
        :param source: 文件路径或 URL
        :return: 清洗后的 Markdown 文本
        """
        pass
