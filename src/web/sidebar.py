"""Workspace and model settings sidebar."""

import os
from typing import Any

import streamlit as st

from src.web.session import persist_conversation, reset_conversation


def activate_profile(profile_id: str) -> None:
    selected = st.session_state.profile_store.set_active(profile_id)
    previous = st.session_state.deep_profile
    previous_id = previous.profile_id if previous is not None else None
    st.session_state.deep_profile = selected
    if profile_id != previous_id:
        st.session_state.last_article = None
        persist_conversation()


def render_sidebar(profile_by_id: dict[str, Any]) -> None:
    with st.sidebar:
        st.header("工作区")
        if st.session_state.conversation_persistence_error:
            st.error(f"对话持久化不可用：{st.session_state.conversation_persistence_error}")

        if len(profile_by_id) > 1:
            pending = st.session_state.pop("_pending_profile_selection", None)
            current = (
                st.session_state.deep_profile.profile_id
                if st.session_state.deep_profile
                and st.session_state.deep_profile.profile_id in profile_by_id
                else next(iter(profile_by_id))
            )
            selected = pending if pending in profile_by_id else current
            if (
                "active_profile_selector" not in st.session_state
                or st.session_state.active_profile_selector not in profile_by_id
                or pending is not None
            ):
                st.session_state.active_profile_selector = selected
            st.selectbox(
                "当前文风画像",
                options=list(profile_by_id),
                format_func=lambda profile_id: f"{profile_by_id[profile_id].name} · {profile_id[:12]}",
                key="active_profile_selector",
                on_change=lambda: activate_profile(st.session_state.active_profile_selector),
            )
        elif profile_by_id:
            profile = next(iter(profile_by_id.values()))
            st.caption(f"当前文风画像：{profile.name} · {profile.profile_id[:12]}")
        else:
            st.caption("尚未建立文风画像")

        active_id = st.session_state.deep_profile.profile_id if st.session_state.deep_profile else None
        memory_stats = (
            st.session_state.memory_mgr.get_memory_stats(profile_id=active_id)
            if active_id
            else {"total_chunks": 0, "sources": []}
        )
        st.caption(f"风格记忆：{memory_stats['total_chunks']} 个切片")

        st.divider()
        st.subheader("本次对话文件")
        if st.session_state.documents:
            for document in st.session_state.documents:
                st.write(f"• {document.get('title', '未命名文件')}")
            if st.button("清空文件", icon=":material/delete:", use_container_width=True):
                st.session_state.documents = []
                st.session_state.document_hashes = set()
                persist_conversation()
                st.rerun()
        else:
            st.caption("无")

        if st.button("新建对话", icon=":material/add_comment:", use_container_width=True):
            reset_conversation()
            st.rerun()

        with st.expander("模型与 Agent 设置"):
            config = st.session_state.config
            api_key = st.text_input(
                "API Key",
                value=config.llm.api_key or os.getenv("OPENAI_API_KEY", ""),
                type="password",
            )
            base_url = st.text_input("Base URL", value=config.llm.base_url)
            model_name = st.text_input("模型", value=config.llm.model)
            st.caption("思考强度可在输入框左下方调整")
            quality_threshold = st.slider(
                "质检合格分",
                min_value=60.0,
                max_value=95.0,
                value=config.agent.quality_threshold,
                step=5.0,
            )
            max_reflections = st.slider(
                "最大反思轮次",
                min_value=0,
                max_value=3,
                value=config.agent.max_reflections,
            )
            st.checkbox("显示 Agent 诊断日志", key="show_agent_logs")
            pdf_options = ["mineru", "markitdown"]
            configured_pdf = (config.extractor.pdf_engine or "mineru").lower()
            pdf_engine = st.selectbox(
                "PDF 解析引擎",
                options=pdf_options,
                index=pdf_options.index(configured_pdf) if configured_pdf in pdf_options else 0,
            )

            config.llm.api_key = api_key
            config.llm.base_url = base_url
            config.llm.model = model_name
            config.agent.quality_threshold = quality_threshold
            config.agent.max_reflections = max_reflections
            config.extractor.pdf_engine = pdf_engine
            st.session_state.coordinator.critic_agent.quality_threshold = quality_threshold
