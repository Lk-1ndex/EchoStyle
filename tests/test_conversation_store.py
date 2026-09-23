import json

import pytest

from src.memory.conversation_store import ConversationConflictError, ConversationStore


def test_conversation_store_round_trip(tmp_path):
    storage_path = tmp_path / "conversation.json"
    store = ConversationStore(str(storage_path))
    state = {
        "messages": [
            {"role": "user", "content": "继续分析附件", "attachments": ["paper.pdf"]},
            {"role": "assistant", "content": "结论如下", "article": "draft"},
        ],
        "documents": [
            {
                "title": "paper",
                "content": "持久化的解析正文",
                "document_id": "doc-hash",
                "engine_used": "pdf_mineru",
            }
        ],
        "document_hashes": ["doc-hash", "doc-hash"],
        "last_article": "draft",
        "context_summary": "用户要求保留论文结论。",
        "summary_covered_messages": 1,
        "context_compression_count": 2,
        "context_notice": "已自动压缩较早的对话",
        "context_report": {"verified": True, "anchor_coverage": 1.0},
    }

    store.save(state)
    restored = ConversationStore(str(storage_path)).load()

    assert restored is not None
    assert restored["messages"] == state["messages"]
    assert restored["documents"] == state["documents"]
    assert restored["document_hashes"] == ["doc-hash"]
    assert restored["context_summary"] == state["context_summary"]
    assert restored["summary_covered_messages"] == 1
    assert restored["context_report"]["verified"] is True


def test_conversation_store_preserves_corrupt_snapshot(tmp_path):
    storage_path = tmp_path / "conversation.json"
    storage_path.write_text("{broken json", encoding="utf-8")
    store = ConversationStore(str(storage_path))

    with pytest.raises(json.JSONDecodeError):
        store.load()

    assert storage_path.read_text(encoding="utf-8") == "{broken json"


def test_failed_save_keeps_previous_snapshot(tmp_path):
    storage_path = tmp_path / "conversation.json"
    store = ConversationStore(str(storage_path))
    original = ConversationStore.empty_state()
    original["messages"] = [{"role": "user", "content": "keep me"}]
    store.save(original)

    invalid = ConversationStore.empty_state()
    invalid["documents"] = [{"title": "bad", "content": object()}]
    with pytest.raises(TypeError):
        store.save(invalid)

    restored = store.load()
    assert restored is not None
    assert restored["messages"] == original["messages"]


def test_v2_appends_messages_and_deduplicates_blobs(tmp_path):
    legacy_path = tmp_path / "conversation_state_v1.json"
    store = ConversationStore(str(legacy_path))
    state = ConversationStore.empty_state()
    shared = "same immutable body"
    state["messages"] = [{"role": "user", "content": shared}]
    state["documents"] = [{"title": "one", "content": shared, "document_id": "one"}]
    store.save(state)

    message_path = next((store.storage_dir / "generations").rglob("*.json"))
    first_mtime = message_path.stat().st_mtime_ns
    state["messages"].append({"role": "assistant", "content": "second message"})
    store.save(state)

    assert message_path.stat().st_mtime_ns == first_mtime
    assert len(list((store.storage_dir / "blobs").glob("*.txt"))) == 2
    assert len(list((store.storage_dir / "generations").rglob("*.json"))) == 2
    assert store.load() == state


def test_v1_migration_preserves_backup_and_is_idempotent(tmp_path):
    legacy_path = tmp_path / "conversation_state_v1.json"
    state = ConversationStore.empty_state()
    state["messages"] = [{"role": "user", "content": "legacy"}]
    legacy_path.write_text(json.dumps({"schema_version": 1, "state": state}), encoding="utf-8")

    first = ConversationStore(str(legacy_path))
    assert first.load() == state
    backup = legacy_path.with_suffix(".json.migrated.bak")
    assert backup.exists()
    assert not legacy_path.exists()

    manifest_before = first.manifest_path.read_bytes()
    second = ConversationStore(str(legacy_path))
    assert second.load() == state
    assert first.manifest_path.read_bytes() == manifest_before


def test_interleaved_stale_store_cannot_overwrite_new_history(tmp_path):
    path = str(tmp_path / "conversation.json")
    first = ConversationStore(path)
    base = ConversationStore.empty_state()
    base["messages"] = [{"role": "user", "content": "base"}]
    first.save(base)

    stale = ConversationStore(path)
    stale_state = stale.load()
    current = first.load()
    current["messages"].append({"role": "assistant", "content": "new"})
    first.save(current)

    stale_state["messages"][0]["content"] = "overwrite"
    with pytest.raises(ConversationConflictError):
        stale.save(stale_state)
    assert first.load()["messages"][-1]["content"] == "new"


def test_clear_switches_generation_without_deleting_old_records(tmp_path):
    store = ConversationStore(str(tmp_path / "conversation.json"))
    state = ConversationStore.empty_state()
    state["messages"] = [{"role": "user", "content": "old"}]
    store.save(state)
    old_manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    old_record = store.storage_dir / old_manifest["message_records"][0]

    store.clear()
    new_manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))

    assert new_manifest["generation_id"] != old_manifest["generation_id"]
    assert old_record.exists()
    assert store.load() == ConversationStore.empty_state()


@pytest.mark.parametrize(
    ("summary", "covered"),
    [("", 1), ("已有摘要", 3)],
)
def test_conversation_store_rejects_invalid_summary_progress(tmp_path, summary, covered):
    state = ConversationStore.empty_state()
    state["messages"] = [{"role": "user", "content": "one"}, {"role": "assistant", "content": "two"}]
    state["context_summary"] = summary
    state["summary_covered_messages"] = covered

    with pytest.raises(ValueError, match="summary_covered_messages"):
        ConversationStore(str(tmp_path / "conversation.json")).save(state)
