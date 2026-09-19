import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from filelock import FileLock


class ConversationStore:
    """Atomic local persistence for the active single-user conversation."""

    SCHEMA_VERSION = 1

    def __init__(self, storage_path: str = "./profiles/conversation_state_v1.json"):
        self.storage_path = Path(storage_path)
        self._file_lock = FileLock(f"{self.storage_path}.lock", timeout=60)

    @staticmethod
    def empty_state() -> Dict[str, Any]:
        return {
            "messages": [],
            "documents": [],
            "document_hashes": [],
            "last_article": None,
            "context_summary": "",
            "summary_covered_messages": 0,
            "context_compression_count": 0,
            "context_notice": "",
            "context_report": None,
        }

    def load(self) -> Optional[Dict[str, Any]]:
        """Return the persisted state, or ``None`` when no snapshot exists."""
        if not self.storage_path.exists():
            return None

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            with open(self.storage_path, "r", encoding="utf-8") as file:
                payload = json.load(file)

        if not isinstance(payload, dict):
            raise ValueError(f"对话存储格式损坏: {self.storage_path}")
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"不支持的对话存储版本: {payload.get('schema_version')}")
        return self._normalize_state(payload.get("state"))

    def save(self, state: Dict[str, Any]) -> None:
        normalized = self._normalize_state(state)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": normalized,
        }
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock:
            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.storage_path.parent,
                    suffix=".tmp",
                    delete=False,
                ) as file:
                    temp_path = Path(file.name)
                    json.dump(payload, file, ensure_ascii=False, indent=2)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temp_path, self.storage_path)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

    def clear(self) -> None:
        self.save(self.empty_state())

    @classmethod
    def _normalize_state(cls, raw_state: Any) -> Dict[str, Any]:
        if not isinstance(raw_state, dict):
            raise ValueError("对话存储缺少 state 对象")

        state = cls.empty_state()
        messages = raw_state.get("messages", state["messages"])
        documents = raw_state.get("documents", state["documents"])
        hashes = raw_state.get("document_hashes", state["document_hashes"])
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise ValueError("对话存储中的 messages 必须是对象列表")
        if not isinstance(documents, list) or any(not isinstance(item, dict) for item in documents):
            raise ValueError("对话存储中的 documents 必须是对象列表")
        if not isinstance(hashes, list) or any(not isinstance(item, str) for item in hashes):
            raise ValueError("对话存储中的 document_hashes 必须是字符串列表")

        summary = raw_state.get("context_summary", "")
        covered = raw_state.get("summary_covered_messages", 0)
        compression_count = raw_state.get("context_compression_count", 0)
        notice = raw_state.get("context_notice", "")
        last_article = raw_state.get("last_article")
        report = raw_state.get("context_report")
        if not isinstance(summary, str) or not isinstance(notice, str):
            raise ValueError("对话摘要和提示必须是字符串")
        if isinstance(covered, bool) or not isinstance(covered, int) or covered < 0:
            raise ValueError("summary_covered_messages 必须是非负整数")
        if covered > len(messages):
            raise ValueError("summary_covered_messages 不能超过消息数量")
        if not summary and covered:
            raise ValueError("没有摘要时 summary_covered_messages 必须为 0")
        if isinstance(compression_count, bool) or not isinstance(compression_count, int) or compression_count < 0:
            raise ValueError("context_compression_count 必须是非负整数")
        if last_article is not None and not isinstance(last_article, str):
            raise ValueError("last_article 必须是字符串或 null")
        if report is not None and not isinstance(report, dict):
            raise ValueError("context_report 必须是对象或 null")

        state.update(
            {
                "messages": copy.deepcopy(messages),
                "documents": copy.deepcopy(documents),
                "document_hashes": sorted(set(hashes)),
                "last_article": last_article,
                "context_summary": summary,
                "summary_covered_messages": covered,
                "context_compression_count": compression_count,
                "context_notice": notice,
                "context_report": copy.deepcopy(report),
            }
        )

        return state
