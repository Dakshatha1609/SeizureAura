# config/config.py

import os
import streamlit as st

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
