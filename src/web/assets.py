"""Web asset loading kept separate from Streamlit page assembly."""

import base64
from pathlib import Path

import streamlit as st


WEB_DIR = Path(__file__).resolve().parent


def brand_mark() -> str:
    asset_path = WEB_DIR / "assets" / "echostyle-logo.png"
    if asset_path.exists():
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        return f'<img class="echo-logo" src="data:image/png;base64,{encoded}" alt="EchoStyle">'
    return '<span class="echo-mark" aria-hidden="true">E</span>'


def inject_styles() -> None:
    styles = (WEB_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
