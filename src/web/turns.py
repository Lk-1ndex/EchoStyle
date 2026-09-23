"""One-turn extraction, context preparation, agent execution, and persistence."""

from typing import Any

import streamlit as st

from src.core.exceptions import IncompleteGenerationError
from src.core.text import count_visible_chars
from src.web.composer import parse_chat_input, profile_context_text
from src.web.ingestion import (
    commit_successful_outcomes,
    extract_uploaded_files,
    extract_wechat_urls,
    failure_messages,
)
from src.web.messages import render_message, store_assistant_result
from src.web.session import persist_conversation


def handle_submission(chat_value: Any, skip_submission: bool, welcome_slot: Any) -> None:
    if chat_value is None or skip_submission:
        return
    if welcome_slot is not None:
        welcome_slot.empty()
    user_text, uploaded_files = parse_chat_input(chat_value)
    attachment_names = [uploaded_file.name for uploaded_file in uploaded_files]
    user_message = {"role": "user", "content": user_text.strip(), "attachments": attachment_names}
    st.session_state.messages.append(user_message)
    render_message(user_message, len(st.session_state.messages) - 1)

    extraction_logs: list[str] = []
    extraction_error = ""
    partial_failures: list[str] = []
    if uploaded_files or "mp.weixin.qq.com" in user_text:
        with st.status("正在读取附件", expanded=True) as extraction_status:
            try:
                upload_batch, upload_logs, skipped_files = extract_uploaded_files(
                    uploaded_files,
                    st.session_state.coordinator,
                    st.session_state.document_hashes,
                )
                url_batch, url_logs = extract_wechat_urls(
                    user_text,
                    st.session_state.coordinator,
                    st.session_state.document_hashes,
                )
                extraction_logs.extend(upload_logs)
                extraction_logs.extend(url_logs)
                new_documents = commit_successful_outcomes(
                    st.session_state.documents,
                    st.session_state.document_hashes,
                    upload_batch,
                    url_batch,
                )
                partial_failures = failure_messages(upload_batch, url_batch)
                for failure in partial_failures:
                    st.warning(f"文件解析失败：{failure}")
                if new_documents:
                    extraction_status.update(
                        label=f"已读取 {len(new_documents)} 份新文件",
                        state="complete",
                        expanded=False,
                    )
                elif skipped_files:
                    extraction_status.update(label="文件已在当前对话中", state="complete", expanded=False)
                elif partial_failures:
                    extraction_status.update(label="附件读取失败", state="error", expanded=True)
                else:
                    extraction_status.update(label="没有发现新文件", state="complete", expanded=False)
            except Exception as exc:
                extraction_error = str(exc)
                extraction_status.update(label="附件读取失败", state="error", expanded=True)
                st.error(extraction_error)

    persist_conversation()
    all_failure_text = "；".join(partial_failures)
    if (extraction_error or all_failure_text) and not user_text.strip() and not st.session_state.documents:
        detail = extraction_error or all_failure_text
        assistant_message = {
            "role": "assistant",
            "content": f"文件解析失败：{detail}",
            "logs": extraction_logs,
        }
    else:
        with st.spinner("Agent 正在处理"):
            try:
                prepared_history, prepared_summary, _, auto_compressed = (
                    st.session_state.context_manager.prepare_history(
                        st.session_state.messages,
                        summary=st.session_state.context_summary,
                        documents=st.session_state.documents,
                        profile_text=profile_context_text(st.session_state.deep_profile),
                    )
                )
                st.session_state.context_summary = prepared_summary
                if auto_compressed:
                    st.session_state.context_compression_count += 1
                    st.session_state.context_notice = "已自动压缩较早的对话"
                    report = st.session_state.context_manager.last_report
                    if report is not None:
                        st.session_state.context_report = {
                            "method": report.method,
                            "source_messages": report.source_messages,
                            "retained_messages": report.retained_messages,
                            "summary_tokens": report.summary_tokens,
                            "anchor_count": report.anchor_count,
                            "anchor_coverage": report.anchor_coverage,
                            "verified": report.verified,
                        }
                        extraction_logs.append(
                            f"[CONTEXT] 自动压缩 ({report.method})，保留最近 {report.retained_messages} 条消息，"
                            f"约束覆盖率 {report.anchor_coverage:.0%}"
                        )
                    else:
                        extraction_logs.append("[CONTEXT] 已自动压缩较早对话，保留最近几轮原文")
                    persist_conversation()

                history_for_agent = prepared_history
                if history_for_agent and history_for_agent[-1] is user_message:
                    history_for_agent = history_for_agent[:-1]
                result = st.session_state.conversation_agent.respond(
                    message=user_text,
                    documents=st.session_state.documents,
                    profile=st.session_state.deep_profile,
                    history=history_for_agent,
                    last_article=st.session_state.last_article,
                    context_summary=st.session_state.context_summary,
                )
                result.logs = extraction_logs + result.logs
                failure_detail = extraction_error or all_failure_text
                if failure_detail:
                    result.content = f"部分附件读取失败：{failure_detail}\n\n{result.content}"
                if result.profile is not None:
                    current_id = (
                        st.session_state.deep_profile.profile_id
                        if st.session_state.deep_profile is not None
                        else None
                    )
                    if not st.session_state.profile_store.has_profile(result.profile.profile_id):
                        st.session_state.profile_store.save(result.profile)
                    st.session_state.deep_profile = result.profile
                    if result.profile.profile_id != current_id:
                        st.session_state._pending_profile_selection = result.profile.profile_id
                if result.article is not None:
                    st.session_state.last_article = result.article
                assistant_message = store_assistant_result(result)
            except Exception as exc:
                if isinstance(exc, IncompleteGenerationError):
                    partial_text = exc.partial_text
                    st.session_state.last_article = partial_text or st.session_state.last_article
                    assistant_message = {
                        "role": "assistant",
                        "content": (
                            f"写作未完成：{exc} 已保留约 {count_visible_chars(partial_text)} / "
                            f"{exc.target_chars} 字的正文。\n\n{partial_text}"
                        ),
                        "article": partial_text or None,
                        "incomplete": True,
                        "logs": extraction_logs,
                    }
                else:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"处理失败：{exc}",
                        "logs": extraction_logs,
                    }

    st.session_state.messages.append(assistant_message)
    persist_conversation()
    st.session_state["_skip_chat_submission"] = True
    st.session_state["_scroll_to_latest_reply"] = True
    st.rerun()
