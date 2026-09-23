"""Composer context and thinking controls."""

from html import escape
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.core.context_manager import ContextSnapshot
from src.extractors.capabilities import SUPPORTED_UPLOAD_TYPES
from src.web.constants import (
    THINKING_EFFORT_LABELS,
    THINKING_EFFORT_OPTIONS,
    THINKING_MODEL_LABELS,
)


def parse_chat_input(value: Any) -> tuple[str, list[Any]]:
    if isinstance(value, str):
        return value, []
    return str(getattr(value, "text", "") or ""), list(getattr(value, "files", []) or [])


def format_token_count(value: int) -> str:
    if value >= 1_000_000:
        number = value / 1_000_000
        return f"{number:.1f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        number = value / 1_000
        return f"{number:.1f}".rstrip("0").rstrip(".") + "K"
    return str(max(0, value))


def profile_context_text(profile: Any) -> str:
    if profile is None:
        return ""
    try:
        return profile.model_dump_json(exclude_none=True)
    except AttributeError:
        return str(profile)


def context_snapshot() -> ContextSnapshot:
    return st.session_state.context_manager.estimate_effective(
        st.session_state.messages,
        documents=st.session_state.documents,
        profile_text=profile_context_text(st.session_state.deep_profile),
        summary=st.session_state.context_summary,
    )


def thinking_model_label(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if normalized in THINKING_MODEL_LABELS:
        return THINKING_MODEL_LABELS[normalized]
    return str(model or "Model").strip() or "Model"


def reset_thinking_effort() -> None:
    st.session_state.composer_thinking_effort = "auto"


def render_live_bridge() -> None:
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
              "--thinking-trigger-width", `${trigger.getBoundingClientRect().width}px`
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
              rail.style.setProperty(`--thinking-dot-${dot}`, dot <= index ? "#79c2f5" : "#c5d1d9");
            }
            const heading = hostDocument.querySelector(".echo-thinking-level");
            if (heading && heading.firstChild) heading.firstChild.nodeValue = `${label} `;
            const trigger = hostDocument.querySelector(
              ".st-key-thinking_effort_popover [data-testid='stPopoverButton']"
            );
            const triggerLabel = trigger?.querySelector("[data-testid='stMarkdownContainer'] p");
            const modelLabel = hostDocument.querySelector(".echo-thinking-model")?.textContent.trim();
            if (triggerLabel) triggerLabel.textContent = modelLabel ? `${modelLabel} ${label}` : label;
            const titledLabel = trigger?.querySelector("[title]");
            if (titledLabel) titledLabel.setAttribute("title", label);
            window.requestAnimationFrame(syncTriggerWidth);
          };
          const bind = () => {
            syncTriggerWidth();
            const thumb = hostDocument.querySelector(
              ".st-key-composer_thinking_effort [role='group'] > div[data-orientation='horizontal'] > div:nth-child(2)"
            );
            if (!thumb || thumb === observedThumb) return;
            sliderObserver?.disconnect();
            observedThumb = thumb;
            sliderObserver = new MutationObserver(() => sync(thumb));
            sliderObserver.observe(thumb, { attributes: true, attributeFilter: ["style"] });
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


def render_chat_input() -> tuple[Any, bool]:
    skip_submission = st.session_state.pop("_skip_chat_submission", False)
    chat_value = st.chat_input(
        "Do anything",
        accept_file="multiple",
        file_type=list(SUPPORTED_UPLOAD_TYPES),
        key="conversation_input",
    )
    return chat_value, skip_submission


def render_toolbar() -> None:
    if "composer_thinking_effort" not in st.session_state:
        configured = st.session_state.config.llm.thinking_effort
        st.session_state.composer_thinking_effort = (
            configured if configured in THINKING_EFFORT_OPTIONS else "auto"
        )

    snapshot = context_snapshot()
    context_limit = max(1, snapshot.soft_limit)
    context_percent = min(100, round(snapshot.used_tokens / context_limit * 100))
    context_left_percent = max(0, 100 - context_percent)
    used_label = format_token_count(snapshot.used_tokens)
    limit_label = format_token_count(context_limit)
    accessible = (
        f"Context window: {context_percent}% used, {context_left_percent}% left; "
        f"{used_label} of {limit_label} tokens used"
    )

    with st.container(key="composer_toolbar"):
        context_column, thinking_column = st.columns(
            [1.0, 0.22], gap="small", vertical_alignment="center"
        )
        with context_column:
            st.markdown(
                f'<div class="echo-context-control" tabindex="0" aria-label="{accessible}">'
                f'<span class="echo-context-ring" style="--context-used: {context_percent}%" aria-hidden="true"></span>'
                '<div class="echo-context-tooltip" role="tooltip"><span>Context window:</span>'
                f'<span>{context_percent}% used ({context_left_percent}% left)</span>'
                f'<span>{used_label} / {limit_label} tokens used</span></div></div>',
                unsafe_allow_html=True,
            )
        with thinking_column:
            current = st.session_state.composer_thinking_effort
            model_label = thinking_model_label(st.session_state.config.llm.model)
            trigger_label = f"{model_label} {THINKING_EFFORT_LABELS[current]}"
            index = THINKING_EFFORT_OPTIONS.index(current)
            progress = index / max(1, len(THINKING_EFFORT_OPTIONS) - 1)
            dot_colors = "; ".join(
                f"--thinking-dot-{dot}: {'#79c2f5' if dot <= index else '#c5d1d9'}"
                for dot in range(len(THINKING_EFFORT_OPTIONS))
            )
            with st.popover(trigger_label, type="secondary", width=280, key="thinking_effort_popover"):
                header, reset = st.columns([1.0, 0.16], gap="small", vertical_alignment="center")
                with header:
                    st.markdown(
                        f'<div class="echo-thinking-heading"><div class="echo-thinking-level">'
                        f'{THINKING_EFFORT_LABELS[current]} <span aria-hidden="true">›</span></div>'
                        f'<div class="echo-thinking-model">{escape(model_label)}</div></div>',
                        unsafe_allow_html=True,
                    )
                with reset:
                    st.button(
                        "重置",
                        icon=":material/refresh:",
                        key="thinking_effort_reset",
                        type="tertiary",
                        help="恢复 Auto",
                        on_click=reset_thinking_effort,
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
                    f'<style>.st-key-composer_thinking_effort [data-testid="stSlider"] '
                    f'[data-orientation="horizontal"] {{ --thinking-progress: {progress:.0%}; {dot_colors}; }}</style>',
                    unsafe_allow_html=True,
                )
    st.session_state.config.llm.thinking_effort = thinking_effort
    with st.container(key="thinking_effort_live_bridge"):
        render_live_bridge()
