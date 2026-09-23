"""Streamlit service initialization and conversation persistence."""

import importlib
from typing import Any

import streamlit as st

from src.agents.conversation_agent import ConversationAgent
from src.agents.coordinator import CoordinatorAgent
from src.core.config import load_config
from src.core.context_manager import ContextManager
from src.memory.conversation_store import ConversationStore
from src.memory.memory_manager import MemoryManager
from src.memory.profile_store import ProfileStore


def summary_covered_messages(manager: Any) -> int:
    return int(
        getattr(
            manager,
            "summary_covered_messages",
            getattr(manager, "_summary_covered_messages", 0),
        )
        or 0
    )


def conversation_payload() -> dict[str, Any]:
    manager: ContextManager = st.session_state.context_manager
    return {
        "messages": list(st.session_state.messages),
        "documents": list(st.session_state.documents),
        "document_hashes": sorted(st.session_state.document_hashes),
        "last_article": st.session_state.last_article,
        "context_summary": st.session_state.context_summary,
        "summary_covered_messages": summary_covered_messages(manager),
        "context_compression_count": st.session_state.context_compression_count,
        "context_notice": st.session_state.context_notice,
        "context_report": st.session_state.context_report,
    }


def persist_conversation() -> None:
    if not st.session_state.get("_conversation_persistence_enabled", True):
        return
    try:
        st.session_state.conversation_store.save(conversation_payload())
        st.session_state.conversation_persistence_error = ""
    except Exception as exc:
        st.session_state.conversation_persistence_error = str(exc)


def restore_conversation() -> None:
    defaults = ConversationStore.empty_state()
    snapshot = None
    st.session_state._conversation_persistence_enabled = True
    st.session_state.conversation_persistence_error = ""
    try:
        snapshot = st.session_state.conversation_store.load()
    except Exception as exc:
        st.session_state._conversation_persistence_enabled = False
        st.session_state.conversation_persistence_error = str(exc)

    source = snapshot or defaults
    for key in (
        "messages",
        "documents",
        "last_article",
        "context_summary",
        "context_compression_count",
        "context_notice",
        "context_report",
    ):
        if snapshot is not None or key not in st.session_state:
            st.session_state[key] = source[key]

    if snapshot is not None or "document_hashes" not in st.session_state:
        hashes = set(source["document_hashes"])
        hashes.update(
            str(document.get("document_id"))
            for document in st.session_state.documents
            if document.get("document_id")
        )
        st.session_state.document_hashes = hashes

    covered = source["summary_covered_messages"] if snapshot is not None else summary_covered_messages(
        st.session_state.context_manager
    )
    st.session_state.context_manager.restore_summary_state(st.session_state.context_summary, covered)
    st.session_state._conversation_state_loaded = True
    st.session_state._scroll_to_latest_reply = bool(st.session_state.messages)
    if snapshot is None and (
        st.session_state.messages
        or st.session_state.documents
        or st.session_state.context_summary
        or st.session_state.last_article
    ):
        persist_conversation()


def reset_conversation() -> None:
    defaults = ConversationStore.empty_state()
    st.session_state.messages = defaults["messages"]
    st.session_state.documents = defaults["documents"]
    st.session_state.document_hashes = set()
    st.session_state.last_article = defaults["last_article"]
    st.session_state.context_summary = defaults["context_summary"]
    st.session_state.context_compression_count = defaults["context_compression_count"]
    st.session_state.context_notice = defaults["context_notice"]
    st.session_state.context_report = defaults["context_report"]
    st.session_state.context_manager.restore_summary_state("", 0)
    st.session_state.conversation_store.clear()
    st.session_state.conversation_persistence_error = ""


def initialize_services() -> None:
    global ContextManager
    if "config" not in st.session_state:
        st.session_state.config = load_config()
    if "profile_store" not in st.session_state:
        st.session_state.profile_store = ProfileStore()
    if "conversation_store" not in st.session_state:
        st.session_state.conversation_store = ConversationStore()
    if "memory_mgr" not in st.session_state:
        st.session_state.memory_mgr = MemoryManager(
            embedding_config=st.session_state.config.embedding,
            llm_config=st.session_state.config.llm,
        )
    if (
        "coordinator" not in st.session_state
        or getattr(st.session_state.coordinator, "profile_store", None)
        is not st.session_state.profile_store
    ):
        st.session_state.coordinator = CoordinatorAgent(
            st.session_state.config,
            memory_manager=st.session_state.memory_mgr,
            profile_store=st.session_state.profile_store,
        )
    if (
        "conversation_agent" not in st.session_state
        or getattr(st.session_state.conversation_agent, "coordinator", None)
        is not st.session_state.coordinator
    ):
        st.session_state.conversation_agent = ConversationAgent(
            st.session_state.config,
            st.session_state.coordinator,
        )

    existing_context_manager = st.session_state.get("context_manager")
    if not hasattr(ContextManager, "restore_summary_state"):
        import src.core.context_manager as context_manager_module

        ContextManager = importlib.reload(context_manager_module).ContextManager
    if (
        existing_context_manager is None
        or getattr(existing_context_manager, "model_provider", None)
        is not st.session_state.conversation_agent.model_provider
        or not hasattr(existing_context_manager, "restore_summary_state")
    ):
        covered = summary_covered_messages(existing_context_manager)
        st.session_state.context_manager = ContextManager(
            st.session_state.conversation_agent.model_provider
        )
        existing_summary = str(st.session_state.get("context_summary", "") or "")
        if existing_summary:
            st.session_state.context_manager.restore_summary_state(existing_summary, covered)

    if "deep_profile" not in st.session_state or st.session_state.deep_profile is None:
        st.session_state.deep_profile = st.session_state.profile_store.get_active()
    if "_conversation_state_loaded" not in st.session_state:
        restore_conversation()
    if "show_agent_logs" not in st.session_state:
        st.session_state.show_agent_logs = False
