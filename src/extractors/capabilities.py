"""Canonical input capabilities shared by CLI, agents, and Web UI."""

from pathlib import Path
from typing import Final
from urllib.parse import urlparse

SUPPORTED_FILE_EXTENSIONS: Final[tuple[str, ...]] = (".pdf", ".docx", ".md", ".txt")
SUPPORTED_UPLOAD_TYPES: Final[tuple[str, ...]] = tuple(
    extension.removeprefix(".") for extension in SUPPORTED_FILE_EXTENSIONS
)
LEGACY_REJECTED_EXTENSIONS: Final[tuple[str, ...]] = (".doc",)


def is_supported_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_source(source: str) -> None:
    """Raise a clear error when a source type is unsupported."""
    if is_supported_url(source):
        return

    extension = Path(source).suffix.lower()
    if extension in LEGACY_REJECTED_EXTENSIONS:
        raise ValueError("旧版 .doc 文件暂不支持，请先转换为 .docx。")
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        supported = "、".join(SUPPORTED_FILE_EXTENSIONS)
        raise ValueError(f"不支持的文件类型 {extension or '（无扩展名）'}；仅支持 {supported}。")
