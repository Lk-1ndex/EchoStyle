import base64
import hashlib
from html import escape
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import streamlit as st
import streamlit.components.v1 as components

from src.agents.base import AgentContext
from src.agents.conversation_agent import ConversationAgent, ConversationResult
from src.agents.coordinator import CoordinatorAgent
from src.core.config import load_config
from src.core.context_manager import ContextManager, ContextSnapshot
from src.memory.memory_manager import MemoryManager
from src.memory.profile_store import ProfileStore


st.set_page_config(
    page_title="EchoStyle",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

def _brand_mark() -> str:
    asset_path = Path(__file__).with_name("assets") / "echostyle-logo.png"
    if asset_path.exists():
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f'<img class="echo-logo" src="data:image/png;base64,{encoded}" alt="EchoStyle">'
    return '<span class="echo-mark" aria-hidden="true">E</span>'


BRAND_MARK = _brand_mark()

THINKING_EFFORT_OPTIONS = ["auto", "off", "low", "high", "max"]
THINKING_EFFORT_LABELS = {
    "auto": "Auto",
    "off": "Off",
    "low": "Low",
    "high": "High",
    "max": "Extra High",
}
THINKING_MODEL_LABELS = {
    "deepseek-flash": "DeepSeek Flash",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}


def _thinking_model_label(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized in THINKING_MODEL_LABELS:
        return THINKING_MODEL_LABELS[normalized]
    return str(model or "Model").strip() or "Model"


def _reset_thinking_effort() -> None:
    st.session_state.composer_thinking_effort = "auto"


def _render_thinking_effort_live_bridge() -> None:
    """Keep the custom slider chrome in sync while Streamlit is still dragging."""
    components.html(
        """
        <script>
        (() => {
          const hostDocument = window.parent.document;
          let sliderObserver = null;
          let observedThumb = null;

          const syncTriggerWidth = () => {
            const trigger = hostDocument.querySelector(
              ".st-key-thinking_effort_popover [data-testid='stPopoverButton']"
            );
            if (!trigger) return;
            hostDocument.documentElement.style.setProperty(
              "--thinking-trigger-width",
              `${trigger.getBoundingClientRect().width}px`
            );
          };

          const sync = (thumb) => {
            const input = thumb.querySelector("input[type='range']");
            const rail = thumb.parentElement;
            if (!input || !rail) return;

            const index = Math.max(0, Math.min(4, Number(input.value) || 0));
            const progress = Number.parseFloat(thumb.style.left) || index * 25;
            const label = input.getAttribute("aria-valuetext") || "Auto";
            rail.style.setProperty("--thinking-progress", `${progress}%`);
            for (let dot = 0; dot < 5; dot += 1) {
              rail.style.setProperty(
                `--thinking-dot-${dot}`,
                dot <= index ? "#79c2f5" : "#c5d1d9"
              );
            }

            const heading = hostDocument.querySelector(".echo-thinking-level");
            if (heading && heading.firstChild) {
              heading.firstChild.nodeValue = `${label} `;
            }

            const trigger = hostDocument.querySelector(
              ".st-key-thinking_effort_popover [data-testid='stPopoverButton']"
            );
            const triggerLabel = trigger?.querySelector(
              "[data-testid='stMarkdownContainer'] p"
            );
            const modelLabel = hostDocument
              .querySelector(".echo-thinking-model")
              ?.textContent.trim();
            if (triggerLabel) {
              triggerLabel.textContent = modelLabel ? `${modelLabel} ${label}` : label;
            }
            const titledLabel = trigger?.querySelector("[title]");
            if (titledLabel) titledLabel.setAttribute("title", label);
            window.requestAnimationFrame(syncTriggerWidth);
          };

          const bind = () => {
            syncTriggerWidth();
            const thumb = hostDocument.querySelector(
              ".st-key-composer_thinking_effort [role='group'] " +
              "> div[data-orientation='horizontal'] > div:nth-child(2)"
            );
            if (!thumb || thumb === observedThumb) return;
            sliderObserver?.disconnect();
            observedThumb = thumb;
            sliderObserver = new MutationObserver(() => sync(thumb));
            sliderObserver.observe(thumb, {
              attributes: true,
              attributeFilter: ["style"],
            });
            sync(thumb);
          };

          bind();
          const pageObserver = new MutationObserver(bind);
          pageObserver.observe(hostDocument.body, { childList: true, subtree: true });
          window.addEventListener("unload", () => {
            sliderObserver?.disconnect();
            pageObserver.disconnect();
          });
        })();
        </script>
        """,
        height=0,
        scrolling=False,
    )


def _initial_messages() -> List[Dict[str, Any]]:
    return []


def _format_token_count(value: int) -> str:
    """Format token counts compactly enough for the composer on mobile."""
    if value >= 1_000_000:
        number = value / 1_000_000
        return f"{number:.1f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        number = value / 1_000
        return f"{number:.1f}".rstrip("0").rstrip(".") + "K"
    return str(max(0, value))


def _profile_context_text(profile: Any) -> str:
    if profile is None:
        return ""
    try:
        return profile.model_dump_json(exclude_none=True)
    except AttributeError:
        return str(profile)


def _context_snapshot() -> ContextSnapshot:
    manager: ContextManager = st.session_state.context_manager
    return manager.estimate_effective(
        st.session_state.messages,
        documents=st.session_state.documents,
        profile_text=_profile_context_text(st.session_state.deep_profile),
        summary=st.session_state.context_summary,
    )


def _activate_profile(profile_id: str) -> None:
    """Make a stored profile active from either profile selector."""
    profile_store: ProfileStore = st.session_state.profile_store
    selected_profile = profile_store.set_active(profile_id)
    previous_profile = st.session_state.deep_profile
    previous_profile_id = previous_profile.profile_id if previous_profile is not None else None
    st.session_state.deep_profile = selected_profile
    if profile_id != previous_profile_id:
        st.session_state.last_article = None


def _parse_chat_input(value: Any) -> Tuple[str, List[Any]]:
    if isinstance(value, str):
        return value, []
    return str(getattr(value, "text", "") or ""), list(getattr(value, "files", []) or [])


def _extract_uploaded_files(
    uploaded_files: Sequence[Any],
    coordinator: CoordinatorAgent,
    known_hashes: set[str],
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    pending: List[Tuple[Any, bytes, str]] = []
    skipped: List[str] = []
    for uploaded_file in uploaded_files:
        payload = uploaded_file.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in known_hashes:
            skipped.append(uploaded_file.name)
            continue
        pending.append((uploaded_file, payload, digest))

    if not pending:
        return [], [], skipped

    temp_paths: List[Path] = []
    source_meta: Dict[str, Tuple[str, str]] = {}
    state = AgentContext()
    try:
        sources: List[str] = []
        for uploaded_file, payload, digest in pending:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(payload)
                temp_path = Path(temp_file.name)
            temp_paths.append(temp_path)
            sources.append(str(temp_path))
            source_meta[temp_path.stem] = (Path(uploaded_file.name).stem, digest)

        articles = coordinator.extract_sources(sources, state=state)
        for article in articles:
            original_title, digest = source_meta.get(
                str(article.get("title", "")),
                (str(article.get("title") or "未命名文件"), ""),
            )
            article["title"] = original_title
            if digest:
                article["document_id"] = digest
                known_hashes.add(digest)
        return articles, state.execution_logs, skipped
    finally:
        for temp_path in temp_paths:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def _extract_wechat_urls(
    message: str,
    coordinator: CoordinatorAgent,
    known_hashes: set[str],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    urls = re.findall(r"https?://mp\.weixin\.qq\.com/[^\s)]+", message)
    new_urls = [url for url in urls if hashlib.sha256(url.encode("utf-8")).hexdigest() not in known_hashes]
    if not new_urls:
        return [], []

    state = AgentContext()
    articles = coordinator.extract_sources(new_urls, state=state)
    for url, article in zip(new_urls, articles):
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        article["document_id"] = digest
        known_hashes.add(digest)
    return articles, state.execution_logs


def _render_message(message: Dict[str, Any], index: int) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message.get("content", ""))
        attachments = message.get("attachments") or []
        if attachments:
            st.caption("附件：" + "、".join(attachments))

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
                file_name=f"echostyle_{index}.md",
                mime="text/markdown",
                key=f"download_article_{index}",
                icon=":material/download:",
            )


def _store_assistant_result(result: ConversationResult) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "content": result.content,
        "article": result.article,
        "report": result.report.model_dump(mode="json") if result.report else None,
        "logs": result.logs,
        "action": result.action.value,
    }


if "config" not in st.session_state:
    st.session_state.config = load_config()

if "profile_store" not in st.session_state:
    st.session_state.profile_store = ProfileStore()

if "memory_mgr" not in st.session_state:
    st.session_state.memory_mgr = MemoryManager(
        embedding_config=st.session_state.config.embedding,
        llm_config=st.session_state.config.llm,
    )

if (
    "coordinator" not in st.session_state
    or getattr(st.session_state.coordinator, "profile_store", None) is not st.session_state.profile_store
):
    st.session_state.coordinator = CoordinatorAgent(
        st.session_state.config,
        memory_manager=st.session_state.memory_mgr,
        profile_store=st.session_state.profile_store,
    )

if (
    "conversation_agent" not in st.session_state
    or getattr(st.session_state.conversation_agent, "coordinator", None) is not st.session_state.coordinator
):
    st.session_state.conversation_agent = ConversationAgent(
        st.session_state.config,
        st.session_state.coordinator,
    )

if (
    "context_manager" not in st.session_state
    or getattr(st.session_state.context_manager, "model_provider", None)
    is not st.session_state.conversation_agent.model_provider
):
    st.session_state.context_manager = ContextManager(
        st.session_state.conversation_agent.model_provider
    )

if "deep_profile" not in st.session_state or st.session_state.deep_profile is None:
    st.session_state.deep_profile = st.session_state.profile_store.get_active()

if "documents" not in st.session_state:
    st.session_state.documents = []
if "document_hashes" not in st.session_state:
    st.session_state.document_hashes = set()
if "messages" not in st.session_state:
    st.session_state.messages = _initial_messages()
if "last_article" not in st.session_state:
    st.session_state.last_article = None
if "show_agent_logs" not in st.session_state:
    st.session_state.show_agent_logs = False
if "context_summary" not in st.session_state:
    st.session_state.context_summary = ""
if "context_compression_count" not in st.session_state:
    st.session_state.context_compression_count = 0
if "context_notice" not in st.session_state:
    st.session_state.context_notice = ""
if "context_report" not in st.session_state:
    st.session_state.context_report = None
if "composer_thinking_effort" not in st.session_state:
    configured_effort = st.session_state.config.llm.thinking_effort
    st.session_state.composer_thinking_effort = (
        configured_effort if configured_effort in THINKING_EFFORT_OPTIONS else "auto"
    )

saved_profiles = st.session_state.profile_store.list_profiles()
profile_by_id = {profile.profile_id: profile for profile in saved_profiles if profile.profile_id}


with st.sidebar:
    st.header("工作区")

    if profile_by_id:
        pending_profile_id = st.session_state.pop("_pending_profile_selection", None)
        current_profile_id = (
            st.session_state.deep_profile.profile_id
            if st.session_state.deep_profile and st.session_state.deep_profile.profile_id in profile_by_id
            else next(iter(profile_by_id))
        )
        selected_profile_id = pending_profile_id if pending_profile_id in profile_by_id else current_profile_id
        if (
            "active_profile_selector" not in st.session_state
            or st.session_state.active_profile_selector not in profile_by_id
            or pending_profile_id is not None
        ):
            st.session_state.active_profile_selector = selected_profile_id

        st.selectbox(
            "当前文风画像",
            options=list(profile_by_id),
            format_func=lambda profile_id: f"{profile_by_id[profile_id].name} · {profile_id[:12]}",
            key="active_profile_selector",
            on_change=lambda: _activate_profile(st.session_state.active_profile_selector),
        )
    else:
        st.caption("尚未建立文风画像")

    active_profile_id = st.session_state.deep_profile.profile_id if st.session_state.deep_profile else None
    memory_stats = (
        st.session_state.memory_mgr.get_memory_stats(profile_id=active_profile_id)
        if active_profile_id
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
            st.rerun()
    else:
        st.caption("无")

    if st.button("新建对话", icon=":material/add_comment:", use_container_width=True):
        st.session_state.messages = _initial_messages()
        st.session_state.documents = []
        st.session_state.document_hashes = set()
        st.session_state.last_article = None
        st.session_state.context_summary = ""
        st.session_state.context_compression_count = 0
        st.session_state.context_notice = ""
        st.session_state.context_report = None
        st.rerun()

    with st.expander("模型与 Agent 设置"):
        api_key = st.text_input(
            "API Key",
            value=st.session_state.config.llm.api_key or os.getenv("OPENAI_API_KEY", ""),
            type="password",
        )
        base_url = st.text_input("Base URL", value=st.session_state.config.llm.base_url)
        model_name = st.text_input("模型", value=st.session_state.config.llm.model)
        st.caption("思考强度可在输入框左下方调整")
        quality_threshold = st.slider(
            "质检合格分",
            min_value=60.0,
            max_value=95.0,
            value=st.session_state.config.agent.quality_threshold,
            step=5.0,
        )
        max_reflections = st.slider(
            "最大反思轮次",
            min_value=0,
            max_value=3,
            value=st.session_state.config.agent.max_reflections,
        )
        st.checkbox("显示 Agent 诊断日志", key="show_agent_logs")
        pdf_engine_options = ["mineru", "markitdown"]
        configured_pdf_engine = (st.session_state.config.extractor.pdf_engine or "mineru").lower()
        pdf_engine = st.selectbox(
            "PDF 解析引擎",
            options=pdf_engine_options,
            index=(
                pdf_engine_options.index(configured_pdf_engine)
                if configured_pdf_engine in pdf_engine_options
                else 0
            ),
        )

        st.session_state.config.llm.api_key = api_key
        st.session_state.config.llm.base_url = base_url
        st.session_state.config.llm.model = model_name
        st.session_state.config.agent.quality_threshold = quality_threshold
        st.session_state.config.agent.max_reflections = max_reflections
        st.session_state.config.extractor.pdf_engine = pdf_engine
        st.session_state.coordinator.critic_agent.quality_threshold = quality_threshold


# Streamlit renders fixed composer widgets before sidebar widgets in its
# element tree.  Keep a synchronized, visually hidden profile selector in the
# main tree so profile selection remains deterministic for keyboard users and
# the existing session state contract; the visible selector stays in the
# workspace sidebar.
if profile_by_id:
    current_profile_id = (
        st.session_state.deep_profile.profile_id
        if st.session_state.deep_profile and st.session_state.deep_profile.profile_id in profile_by_id
        else next(iter(profile_by_id))
    )
    st.session_state.compat_profile_selector = current_profile_id
    st.selectbox(
        "当前文风画像",
        options=list(profile_by_id),
        format_func=lambda profile_id: f"{profile_by_id[profile_id].name} · {profile_id[:12]}",
        key="compat_profile_selector",
        label_visibility="collapsed",
        on_change=lambda: _activate_profile(st.session_state.compat_profile_selector),
    )


welcome_slot = st.empty() if not st.session_state.messages else None
if welcome_slot is not None:
    welcome_slot.markdown(
        f'<div class="echo-welcome">{BRAND_MARK}'
        '<h1>What should we write?</h1></div>',
        unsafe_allow_html=True,
    )

for message_index, stored_message in enumerate(st.session_state.messages):
    _render_message(stored_message, message_index)


context_snapshot = _context_snapshot()
context_limit = max(1, context_snapshot.soft_limit)
context_percent = min(100, round(context_snapshot.used_tokens / context_limit * 100))
context_left_percent = max(0, 100 - context_percent)
context_used_label = _format_token_count(context_snapshot.used_tokens)
context_limit_label = _format_token_count(context_limit)
context_accessible_label = (
    f"Context window: {context_percent}% used, {context_left_percent}% left; "
    f"{context_used_label} of {context_limit_label} tokens used"
)

with st.container(key="composer_toolbar"):
    context_column, thinking_column = st.columns(
        [1.0, 0.22], gap="small", vertical_alignment="center"
    )
    with context_column:
        st.markdown(
            f'<div class="echo-context-control" tabindex="0" '
            f'aria-label="{context_accessible_label}">'
            f'<span class="echo-context-ring" style="--context-used: {context_percent}%" '
            f'aria-hidden="true"></span>'
            '<div class="echo-context-tooltip" role="tooltip">'
            '<span>Context window:</span>'
            f'<span>{context_percent}% used ({context_left_percent}% left)</span>'
            f'<span>{context_used_label} / {context_limit_label} tokens used</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with thinking_column:
        current_effort = st.session_state.composer_thinking_effort
        thinking_model_label = _thinking_model_label(st.session_state.config.llm.model)
        thinking_trigger_label = (
            f"{thinking_model_label} {THINKING_EFFORT_LABELS[current_effort]}"
        )
        thinking_index = THINKING_EFFORT_OPTIONS.index(current_effort)
        thinking_progress = thinking_index / max(1, len(THINKING_EFFORT_OPTIONS) - 1)
        thinking_dot_colors = "; ".join(
            f"--thinking-dot-{index}: "
            f"{'#79c2f5' if index <= thinking_index else '#c5d1d9'}"
            for index in range(len(THINKING_EFFORT_OPTIONS))
        )
        with st.popover(
            thinking_trigger_label,
            type="secondary",
            width=280,
            key="thinking_effort_popover",
        ):
            header_column, reset_column = st.columns([1.0, 0.16], gap="small", vertical_alignment="center")
            with header_column:
                st.markdown(
                    f'<div class="echo-thinking-heading">'
                    f'<div class="echo-thinking-level">{THINKING_EFFORT_LABELS[current_effort]} '
                    '<span aria-hidden="true">›</span></div>'
                    f'<div class="echo-thinking-model">{escape(thinking_model_label)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            with reset_column:
                st.button(
                    "重置",
                    icon=":material/refresh:",
                    key="thinking_effort_reset",
                    type="tertiary",
                    help="恢复 Auto",
                    on_click=_reset_thinking_effort,
                )

            thinking_effort = st.select_slider(
                "模型思考强度",
                options=THINKING_EFFORT_OPTIONS,
                format_func=lambda value: THINKING_EFFORT_LABELS[value],
                key="composer_thinking_effort",
                label_visibility="collapsed",
                help="仅对 DeepSeek 官方 API 生效；路由和结构化任务会自动关闭思考。",
            )
            st.markdown(
                f'<style>.st-key-composer_thinking_effort '
                f'[data-testid="stSlider"] [data-orientation="horizontal"] '
                f'{{ --thinking-progress: {thinking_progress:.0%}; '
                f'{thinking_dot_colors}; }}</style>',
                unsafe_allow_html=True,
            )

st.session_state.config.llm.thinking_effort = thinking_effort

with st.container(key="thinking_effort_live_bridge"):
    _render_thinking_effort_live_bridge()


skip_chat_submission = st.session_state.pop("_skip_chat_submission", False)
chat_value = st.chat_input(
    "Do anything",
    accept_file="multiple",
    file_type=["pdf", "docx", "doc", "md", "txt"],
    key="conversation_input",
)

if chat_value is not None and not skip_chat_submission:
    if welcome_slot is not None:
        welcome_slot.empty()
    user_text, uploaded_files = _parse_chat_input(chat_value)
    attachment_names = [uploaded_file.name for uploaded_file in uploaded_files]
    display_text = user_text.strip() or "已上传文件"
    user_message = {"role": "user", "content": display_text, "attachments": attachment_names}
    st.session_state.messages.append(user_message)
    _render_message(user_message, len(st.session_state.messages) - 1)

    extraction_logs: List[str] = []
    extraction_error = ""
    if uploaded_files or "mp.weixin.qq.com" in user_text:
        with st.status("正在读取附件", expanded=True) as extraction_status:
            try:
                new_documents, upload_logs, skipped_files = _extract_uploaded_files(
                    uploaded_files,
                    st.session_state.coordinator,
                    st.session_state.document_hashes,
                )
                url_documents, url_logs = _extract_wechat_urls(
                    user_text,
                    st.session_state.coordinator,
                    st.session_state.document_hashes,
                )
                new_documents.extend(url_documents)
                extraction_logs.extend(upload_logs)
                extraction_logs.extend(url_logs)
                st.session_state.documents.extend(new_documents)
                if new_documents:
                    extraction_status.update(
                        label=f"已读取 {len(new_documents)} 份新文件",
                        state="complete",
                        expanded=False,
                    )
                elif skipped_files:
                    extraction_status.update(label="文件已在当前对话中", state="complete", expanded=False)
                else:
                    extraction_status.update(label="没有发现新文件", state="complete", expanded=False)
            except Exception as exc:
                extraction_error = str(exc)
                extraction_status.update(label="附件读取失败", state="error", expanded=True)
                st.error(extraction_error)

    if extraction_error and not user_text.strip():
        assistant_message = {
            "role": "assistant",
            "content": f"文件解析失败：{extraction_error}",
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
                        profile_text=_profile_context_text(st.session_state.deep_profile),
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

                # The current user turn is passed as ``message`` separately;
                # avoid duplicating it in the history supplied to the agent.
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
                if extraction_error:
                    result.content = f"部分附件读取失败：{extraction_error}\n\n{result.content}"
                if result.profile is not None:
                    current_profile_id = (
                        st.session_state.deep_profile.profile_id
                        if st.session_state.deep_profile is not None
                        else None
                    )
                    if not st.session_state.profile_store.has_profile(result.profile.profile_id):
                        st.session_state.profile_store.save(result.profile)
                    st.session_state.deep_profile = result.profile
                    if result.profile.profile_id != current_profile_id:
                        st.session_state._pending_profile_selection = result.profile.profile_id
                if result.article is not None:
                    st.session_state.last_article = result.article
                assistant_message = _store_assistant_result(result)
            except Exception as exc:
                assistant_message = {
                    "role": "assistant",
                    "content": f"处理失败：{exc}",
                    "logs": extraction_logs,
                }

    st.session_state.messages.append(assistant_message)
    # Rebuild the chat once after a response so Streamlit recalculates the scroll
    # container with the complete latest message instead of leaving it mid-stream.
    st.session_state["_skip_chat_submission"] = True
    st.rerun()
