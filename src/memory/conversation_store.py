import copy
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from filelock import FileLock


class ConversationConflictError(RuntimeError):
    """Raised when an older snapshot would overwrite newer persisted history."""


class ConversationStore:
    """Segmented local persistence for the active single-user conversation."""

    SCHEMA_VERSION = 2
    LEGACY_SCHEMA_VERSION = 1

    def __init__(self, storage_path: str = "./profiles/conversation_state_v1.json"):
        self.storage_path = Path(storage_path)
        stem = self.storage_path.stem
        v2_name = f"{stem[:-3]}_v2" if stem.endswith("_v1") else f"{stem}_v2"
        self.storage_dir = self.storage_path.with_name(v2_name)
        self.manifest_path = self.storage_dir / "manifest.json"
        self._file_lock = FileLock(f"{self.storage_dir}.lock", timeout=60)
        self._known_revision: Optional[int] = None

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
        """Return the persisted state, migrating a valid v1 snapshot once."""
        if not self.manifest_path.exists() and not self.storage_path.exists():
            return None

        self.storage_dir.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            if not self.manifest_path.exists():
                state = self._read_legacy_state_locked()
                manifest = self._commit_state_locked(
                    state,
                    current_manifest=None,
                    force_new_generation=True,
                )
                self._archive_legacy_locked()
            else:
                manifest = self._read_manifest_locked()
                state = self._state_from_manifest_locked(manifest)
            self._known_revision = int(manifest["revision"])
            return state

    def save(self, state: Dict[str, Any]) -> None:
        normalized = self._normalize_state(state)
        # Reject non-JSON state before creating any immutable records.
        json.dumps(normalized, ensure_ascii=False)
        self.storage_dir.parent.mkdir(parents=True, exist_ok=True)

        with self._file_lock:
            current_manifest: Optional[Dict[str, Any]]
            if not self.manifest_path.exists() and self.storage_path.exists():
                legacy_state = self._read_legacy_state_locked()
                migrated_manifest = self._commit_state_locked(
                    legacy_state,
                    current_manifest=None,
                    force_new_generation=True,
                )
                if migrated_manifest is None:
                    raise RuntimeError("旧版对话迁移未生成有效 manifest。")
                current_manifest = migrated_manifest
                self._archive_legacy_locked()
            else:
                current_manifest = (
                    self._read_manifest_locked() if self.manifest_path.exists() else None
                )

            if (
                current_manifest is not None
                and self._known_revision is not None
                and int(current_manifest["revision"]) != self._known_revision
            ):
                current_state = self._state_from_manifest_locked(current_manifest)
                if not self._is_history_prefix(current_state, normalized):
                    raise ConversationConflictError(
                        "对话已被其他进程更新，拒绝用旧快照覆盖较新的消息或文档。"
                    )

            manifest = self._commit_state_locked(
                normalized,
                current_manifest=current_manifest,
                force_new_generation=False,
            )
            self._known_revision = int(manifest["revision"])

    def clear(self) -> None:
        """Atomically switch the manifest to a new empty generation."""
        self.storage_dir.parent.mkdir(parents=True, exist_ok=True)
        with self._file_lock:
            current_manifest = (
                self._read_manifest_locked() if self.manifest_path.exists() else None
            )
            manifest = self._commit_state_locked(
                self.empty_state(),
                current_manifest=current_manifest,
                force_new_generation=True,
            )
            self._known_revision = int(manifest["revision"])

    @staticmethod
    def _is_history_prefix(current: Dict[str, Any], proposed: Dict[str, Any]) -> bool:
        messages = current["messages"]
        documents = current["documents"]
        return (
            proposed["messages"][: len(messages)] == messages
            and proposed["documents"][: len(documents)] == documents
        )

    def _commit_state_locked(
        self,
        state: Dict[str, Any],
        current_manifest: Optional[Dict[str, Any]],
        force_new_generation: bool,
    ) -> Dict[str, Any]:
        if current_manifest is not None and not force_new_generation:
            current_state = self._state_from_manifest_locked(current_manifest)
            if not self._is_history_prefix(current_state, state):
                raise ConversationConflictError(
                    "保存内容不是当前对话的增量；请显式新建对话后再写入。"
                )
            generation_id = str(current_manifest["generation_id"])
            message_records = list(current_manifest["message_records"])
            document_records = list(current_manifest["document_records"])
            new_messages = state["messages"][len(current_state["messages"]) :]
            new_documents = state["documents"][len(current_state["documents"]) :]
        else:
            generation_id = uuid4().hex
            message_records = []
            document_records = []
            new_messages = state["messages"]
            new_documents = state["documents"]

        for message in new_messages:
            encoded = self._encode_message_locked(message)
            index = len(message_records)
            record_hash = self._json_hash(encoded)
            relative = Path("generations") / generation_id / "messages" / f"{index:08d}-{record_hash}.json"
            self._write_json_immutable_locked(self.storage_dir / relative, encoded)
            message_records.append(relative.as_posix())

        for document in new_documents:
            encoded = self._encode_document_locked(document)
            record_hash = self._json_hash(encoded)
            relative = Path("documents") / f"{record_hash}.json"
            self._write_json_immutable_locked(self.storage_dir / relative, encoded)
            document_records.append(relative.as_posix())

        last_article = state["last_article"]
        last_article_blob = (
            self._store_blob_locked(last_article) if isinstance(last_article, str) else None
        )
        revision = int(current_manifest["revision"]) + 1 if current_manifest else 1
        manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "revision": revision,
            "generation_id": generation_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "message_records": message_records,
            "document_records": document_records,
            "document_hashes": state["document_hashes"],
            "last_article_blob": last_article_blob,
            "context_summary": state["context_summary"],
            "summary_covered_messages": state["summary_covered_messages"],
            "context_compression_count": state["context_compression_count"],
            "context_notice": state["context_notice"],
            "context_report": state["context_report"],
        }
        self._atomic_write_json(self.manifest_path, manifest)
        return manifest

    def _read_legacy_state_locked(self) -> Dict[str, Any]:
        with open(self.storage_path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"对话存储格式损坏: {self.storage_path}")
        if payload.get("schema_version") != self.LEGACY_SCHEMA_VERSION:
            raise ValueError(f"不支持的对话存储版本: {payload.get('schema_version')}")
        return self._normalize_state(payload.get("state"))

    def _archive_legacy_locked(self) -> None:
        if not self.storage_path.exists():
            return
        backup = self.storage_path.with_suffix(self.storage_path.suffix + ".migrated.bak")
        if backup.exists():
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = self.storage_path.with_suffix(
                self.storage_path.suffix + f".{timestamp}.migrated.bak"
            )
        os.replace(self.storage_path, backup)

    def _read_manifest_locked(self) -> Dict[str, Any]:
        with open(self.manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict):
            raise ValueError(f"对话 manifest 格式损坏: {self.manifest_path}")
        if manifest.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError(f"不支持的对话存储版本: {manifest.get('schema_version')}")
        if isinstance(manifest.get("revision"), bool) or not isinstance(manifest.get("revision"), int):
            raise ValueError("对话 manifest revision 必须是整数")
        for key in ("message_records", "document_records", "document_hashes"):
            if not isinstance(manifest.get(key), list):
                raise ValueError(f"对话 manifest 缺少列表字段: {key}")
        return manifest

    def _state_from_manifest_locked(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        messages = [
            self._decode_message(self._read_json_record(relative))
            for relative in manifest["message_records"]
        ]
        documents = [
            self._decode_document(self._read_json_record(relative))
            for relative in manifest["document_records"]
        ]
        last_blob = manifest.get("last_article_blob")
        raw_state = {
            "messages": messages,
            "documents": documents,
            "document_hashes": manifest["document_hashes"],
            "last_article": self._read_blob(last_blob) if last_blob else None,
            "context_summary": manifest.get("context_summary", ""),
            "summary_covered_messages": manifest.get("summary_covered_messages", 0),
            "context_compression_count": manifest.get("context_compression_count", 0),
            "context_notice": manifest.get("context_notice", ""),
            "context_report": manifest.get("context_report"),
        }
        return self._normalize_state(raw_state)

    def _read_json_record(self, relative: str) -> Dict[str, Any]:
        path = self._resolve_record_path(relative)
        with open(path, "r", encoding="utf-8") as file:
            payload = json.load(file)
        if not isinstance(payload, dict):
            raise ValueError(f"对话记录格式损坏: {path}")
        return payload

    def _resolve_record_path(self, relative: str) -> Path:
        if not isinstance(relative, str):
            raise ValueError("对话记录引用必须是字符串")
        root = self.storage_dir.resolve()
        path = (self.storage_dir / relative).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"非法对话记录路径: {relative}")
        return path

    def _encode_message_locked(self, message: Dict[str, Any]) -> Dict[str, Any]:
        encoded = copy.deepcopy(message)
        for key in ("content", "article"):
            value = encoded.pop(key, None)
            if isinstance(value, str):
                encoded[f"{key}_blob"] = self._store_blob_locked(value)
            elif value is not None:
                encoded[key] = value
        return encoded

    def _decode_message(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        message = copy.deepcopy(encoded)
        for key in ("content", "article"):
            blob_key = f"{key}_blob"
            if blob_key in message:
                message[key] = self._read_blob(message.pop(blob_key))
        return message

    def _encode_document_locked(self, document: Dict[str, Any]) -> Dict[str, Any]:
        encoded = copy.deepcopy(document)
        content = encoded.pop("content", None)
        if isinstance(content, str):
            encoded["content_blob"] = self._store_blob_locked(content)
        elif content is not None:
            encoded["content"] = content
        return encoded

    def _decode_document(self, encoded: Dict[str, Any]) -> Dict[str, Any]:
        document = copy.deepcopy(encoded)
        if "content_blob" in document:
            document["content"] = self._read_blob(document.pop("content_blob"))
        return document

    def _store_blob_locked(self, text: str) -> str:
        payload = text.encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        path = self.storage_dir / "blobs" / f"{digest}.txt"
        self._write_immutable_bytes_locked(path, payload)
        return digest

    def _read_blob(self, digest: str) -> str:
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"非法内容 blob 引用: {digest}")
        return (self.storage_dir / "blobs" / f"{digest}.txt").read_text(encoding="utf-8")

    @staticmethod
    def _json_hash(payload: Dict[str, Any]) -> str:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _write_json_immutable_locked(self, path: Path, payload: Dict[str, Any]) -> None:
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self._write_immutable_bytes_locked(path, raw)

    def _write_immutable_bytes_locked(self, path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError(f"不可变对话对象发生哈希冲突: {path}")
            return
        self._atomic_write_bytes(path, payload)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        ConversationStore._atomic_write_bytes(path, raw)

    @staticmethod
    def _atomic_write_bytes(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_path = Path(file.name)
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

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
