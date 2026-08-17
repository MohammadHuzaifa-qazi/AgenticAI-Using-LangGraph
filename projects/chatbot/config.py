from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("groq_api_key")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("groq_api_key")

api_key = get_api_key()

if not api_key:
    raise ValueError("groq_api_key not found. Set it in Streamlit Secrets or .env")

model = ChatGroq(model='openai/gpt-oss-20b', api_key=api_key)
