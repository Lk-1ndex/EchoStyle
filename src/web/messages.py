"""Chat transcript serialization and rendering."""

from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from src.agents.conversation_agent import ConversationResult


def render_message(message: dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        content = message.get("content", "")
        attachments = message.get("attachments") or []
        if attachments:
            attachment_class = "echo-attachments echo-attachments-only" if not content else "echo-attachments"
            chips = ""
            for name in attachments:
                suffix = Path(str(name)).suffix.removeprefix(".").upper()
                file_kind = {"DOCX": "DOC", "TEXT": "TXT"}.get(suffix, suffix[:4] or "FILE")
                chips += (
                    '<span class="echo-attachment">'
                    f'<span class="echo-attachment-kind" aria-hidden="true">{escape(file_kind)}</span>'
                    f'<span class="echo-attachment-name">{escape(str(name))}</span>'
                    "</span>"
                )
            st.markdown(f'<div class="{attachment_class}">{chips}</div>', unsafe_allow_html=True)
        if content:
            st.markdown(content)

        report = message.get("report")
        if report:
            with st.expander("EchoEval 评测"):
                columns = st.columns(4)
                columns[0].metric("综合得分", f"{report.get('overall_score', 0):.1f}")
                columns[1].metric("文风相似", f"{report.get('style_fidelity', 0):.1f}")
                columns[2].metric("句式拟合", f"{report.get('stylometric_similarity', 0):.1f}")
                columns[3].metric("去 AI 味", f"{report.get('anti_ai_score', 0):.1f}")
                if report.get("feedback"):
                    st.write(report["feedback"])

        logs = message.get("logs") or []
        if logs and st.session_state.get("show_agent_logs", False):
            with st.expander(f"诊断日志 · {len(logs)} 条"):
                for entry in logs:
                    st.text(entry)

        article = message.get("article")
        if article:
            st.download_button(
                "下载 Markdown",
                data=article,
                file_name=f"echostyle_{'partial_' if message.get('incomplete') else ''}{index}.md",
                mime="text/markdown",
                key=f"download_article_{index}",
                icon=":material/download:",
            )


def store_assistant_result(result: ConversationResult) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content,
        "article": result.article,
        "report": result.report.model_dump(mode="json") if result.report else None,
        "logs": result.logs,
        "action": result.action.value,
    }
