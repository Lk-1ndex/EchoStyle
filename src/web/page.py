"""Ordered Streamlit page assembly."""

from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from src.web.assets import brand_mark
from src.web.composer import render_chat_input, render_toolbar
from src.web.messages import render_message
from src.web.sidebar import render_sidebar
from src.web.turns import handle_submission


def render_scroll_bridge() -> None:
    if not st.session_state.pop("_scroll_to_latest_reply", False):
        return
    components.html(
        """
        <script>
        (() => {
          const host = window.parent.document;
          const expectedCount = __EXPECTED_COUNT__;
          const alignReply = () => {
            const messages = [...host.querySelectorAll('[data-testid="stChatMessage"]')];
            if (messages.length < expectedCount) return;
            const latestUser = messages.reverse().find((message) =>
              message.querySelector('[aria-label="Chat message from user"]')
            );
            const start = latestUser || messages[0];
            if (start) start.scrollIntoView({ block: "start", behavior: "instant" });
          };
          const observer = new MutationObserver(() => window.requestAnimationFrame(alignReply));
          observer.observe(host.body, { childList: true, subtree: true });
          for (const delay of [0, 150, 500, 1200, 2500]) window.setTimeout(alignReply, delay);
          window.setTimeout(() => observer.disconnect(), 2700);
          window.addEventListener("unload", () => observer.disconnect());
        })();
        </script>
        """.replace("__EXPECTED_COUNT__", str(len(st.session_state.messages))),
        height=0,
        scrolling=False,
    )


def render_page() -> None:
    saved_profiles = st.session_state.profile_store.list_profiles()
    profile_by_id: dict[str, Any] = {
        profile.profile_id: profile for profile in saved_profiles if profile.profile_id
    }

    welcome_slot = st.empty() if not st.session_state.messages else None
    if welcome_slot is not None:
        welcome_slot.markdown(
            f'<div class="echo-welcome">{brand_mark()}<h1>What should we write?</h1></div>',
            unsafe_allow_html=True,
        )

    for message_index, stored_message in enumerate(st.session_state.messages):
        render_message(stored_message, message_index)

    chat_value, skip_submission = render_chat_input()
    render_scroll_bridge()
    render_sidebar(profile_by_id)
    render_toolbar()
    handle_submission(chat_value, skip_submission, welcome_slot)
