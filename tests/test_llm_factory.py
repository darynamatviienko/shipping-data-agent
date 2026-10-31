"""Offline tests for the LLM factory (Gemini / Anthropic / OpenAI)."""

from __future__ import annotations

import pytest

from shipping_data_agent.agent import llm as llm_mod
from shipping_data_agent.agent.llm import LlmConfigError, clear_chat_model_cache


@pytest.fixture(autouse=True)
def _clear_cache_and_env(monkeypatch):
    """Isolate from a real developer .env (load_dotenv would refill delenv'd keys)."""
    clear_chat_model_cache()
    # Empty strings beat load_dotenv(override=False); delenv alone is not enough.
    for key in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "ANTHROPIC_MODEL",
        "OPENAI_MODEL",
        "GEMINI_MODEL",
    ):
        monkeypatch.setenv(key, "")
    yield
    clear_chat_model_cache()


def test_llm_configured_false_without_keys():
    assert llm_mod.llm_configured() is False


def test_llm_configured_true_for_gemini_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini")
    assert llm_mod.llm_configured() is True


def test_llm_configured_true_for_google_api_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google")
    assert llm_mod.llm_configured() is True


def test_resolve_provider_explicit_gemini(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "also-set")
    assert llm_mod._resolve_provider() == "gemini"


def test_resolve_provider_auto_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    assert llm_mod._resolve_provider() == "gemini"


def test_get_chat_model_gemini_builds_without_network(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-not-real")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-flash-lite-latest")
    model = llm_mod.get_chat_model()
    assert model.__class__.__name__ == "ChatGoogleGenerativeAI"
    assert getattr(model, "model", None) == "gemini-flash-lite-latest"


def test_get_chat_model_gemini_accepts_google_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    model = llm_mod.get_chat_model()
    assert model.__class__.__name__ == "ChatGoogleGenerativeAI"


def test_get_chat_model_gemini_missing_key_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(LlmConfigError, match="GEMINI_API_KEY"):
        llm_mod.get_chat_model()


def test_openai_and_anthropic_paths_still_selected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai")
    model = llm_mod.get_chat_model()
    assert model.__class__.__name__ == "ChatOpenAI"
    clear_chat_model_cache()

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
    model = llm_mod.get_chat_model()
    assert model.__class__.__name__ == "ChatAnthropic"
