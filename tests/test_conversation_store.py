import json

import pytest

from src.memory.conversation_store import ConversationStore


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
