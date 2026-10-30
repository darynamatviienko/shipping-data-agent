#!/usr/bin/env python3
"""Simple Streamlit chat UI for the shipping LangGraph agent (Step 7)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from shipping_data_agent.agent import ask, llm_configured  # noqa: E402
from shipping_data_agent.agent.llm import LlmConfigError  # noqa: E402
from shipping_data_agent.tools import DEFAULT_DB_PATH  # noqa: E402

SAMPLE_QUESTIONS = [
    "How many shipments were delivered to Germany?",
    "Find invoice discrepancies and summarise the error types",
    "What is the average delivery_days by carrier?",
    "How many invoices are there?",
]


def main() -> None:
    st.set_page_config(
        page_title="Shipping Data Agent",
        page_icon="📦",
        layout="centered",
    )
    st.title("Shipping Data Agent")
    st.caption(
        "Ask natural-language questions about shipments and invoice discrepancies. "
        "Answers come from the DuckDB warehouse via a LangGraph agent."
    )

    if not DEFAULT_DB_PATH.exists():
        st.error(
            f"Warehouse not found at `{DEFAULT_DB_PATH}`. "
            "Run `make data && make etl` first."
        )
        st.stop()

    if not llm_configured():
        st.warning(
            "No LLM API key found. Copy `.env.example` → `.env` and set "
            "`GEMINI_API_KEY` / `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, or "
            "`OPENAI_API_KEY` (plus optional `LLM_PROVIDER`)."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("Sample questions")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, use_container_width=True):
                st.session_state.pending = q
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pending", None)
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.session_state.pop("pending", None) or st.chat_input(
        "Ask about shipments or invoices…"
    )
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm_configured():
            answer = (
                "Configure an API key in `.env` before chatting "
                "(see `.env.example`)."
            )
            st.markdown(answer)
        else:
            with st.spinner("Thinking…"):
                try:
                    answer = ask(prompt)
                except LlmConfigError as exc:
                    answer = f"LLM config error: {exc}"
                except FileNotFoundError as exc:
                    answer = f"{exc}\n\nRun `make etl` first."
                except Exception as exc:  # noqa: BLE001 — surface to UI
                    answer = f"Agent error: {exc}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
