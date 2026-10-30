"""LLM factory — keys and provider come from the environment only."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel


class LlmConfigError(RuntimeError):
    """Raised when no usable LLM provider/key is configured."""


def llm_configured() -> bool:
    """Return True when at least one supported API key is present."""
    load_dotenv()
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY"))


def _resolve_provider() -> str:
    load_dotenv()
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit in {"anthropic", "openai"}:
        return explicit
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "anthropic"


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Build a chat model from env vars. Never hardcodes API keys."""
    load_dotenv()
    provider = _resolve_provider()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LlmConfigError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add a key."
            )
        from langchain_openai import ChatOpenAI

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, api_key=api_key, temperature=0)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LlmConfigError(
            "LLM_PROVIDER=anthropic (default) but ANTHROPIC_API_KEY is not set. "
            "Copy .env.example to .env and add a key, or set LLM_PROVIDER=openai."
        )
    from langchain_anthropic import ChatAnthropic

    model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
    return ChatAnthropic(model=model, api_key=api_key, temperature=0)


def clear_chat_model_cache() -> None:
    """Reset the cached chat model (useful in tests)."""
    get_chat_model.cache_clear()
