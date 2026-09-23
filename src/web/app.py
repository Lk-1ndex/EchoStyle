import streamlit as st

from src.web.assets import inject_styles
from src.web.page import render_page
from src.web.session import initialize_services


st.set_page_config(
    page_title="EchoStyle",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_styles()
initialize_services()
render_page()
