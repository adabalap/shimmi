"""
tests/conftest.py — Shimmi Phase 1

Shared pytest fixtures.  ALL fixtures are offline by default:
  - No network calls
  - No LLM calls (Groq/Gemini mocked)
  - No real ChromaDB (in-memory or fully mocked)
  - SQLite uses tmp_path so it's isolated per-test

Marks:
  @pytest.mark.live    — requires GROQ_API_KEY / GEMINI_API_KEY (skipped in CI)
  @pytest.mark.integration — requires a running Shimmi bot on localhost:6000

Run offline tests:
    pytest tests/unit/

Run integration tests (mocked LLM):
    pytest tests/integration/

Run live tests (costs real quota):
    pytest tests/live/ -m live
"""
from __future__ import annotations

import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Make sure the shimmi_p1 package root is importable when running pytest
# from the project root.  Adjust if your layout differs.
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Canned LLM responses used by mock_llm fixture
# ─────────────────────────────────────────────────────────────────────────────

CANNED_ORCHESTRATOR_ANSWER = json.dumps({
    "action": "answer",
    "reasoning": "User asked a simple question; answering from facts.",
    "text": "Your city is Hyderabad.",
    "query": "",
    "question": "",
    "memory_updates": [],
    "reminders": [],
    "tool_call": None,
})

CANNED_ORCHESTRATOR_SEARCH = json.dumps({
    "action": "search",
    "reasoning": "User wants live weather data.",
    "text": "",
    "query": "weather Hyderabad India",
    "question": "",
    "memory_updates": [],
    "reminders": [],
    "tool_call": {"tool": "weather", "city": "Hyderabad", "country": "IN", "days": 3},
})

CANNED_ORCHESTRATOR_SEARCH_THEN_ANSWER = json.dumps({
    "action": "answer",
    "reasoning": "Search result received; answering now.",
    "text": "The weather in Hyderabad is 32°C and sunny.",
    "query": "",
    "question": "",
    "memory_updates": [],
    "reminders": [],
    "tool_call": None,
})

CANNED_EXTRACT_RESULT = json.dumps({
    "memory_updates": [{"key": "name", "value": "Phani"}],
})

CANNED_VERIFY_RESULT = json.dumps({
    "approved": [{"key": "name", "value": "Phani", "confidence": 1.0}],
})

CANNED_FORMAT_RESULT = json.dumps({
    "text": "Your city is Hyderabad. 🌆",
})

CANNED_EMPTY_EXTRACT = json.dumps({"memory_updates": []})


# ─────────────────────────────────────────────────────────────────────────────
# Event loop fixture (asyncio)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Shared event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# Database fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """
    Isolated SQLiteMemory instance backed by a temp file.
    No ChromaDB — pure SQL, fast.
    """
    # Patch settings so SQLiteMemory uses our tmp path
    from app.database import SQLiteMemory
    store = SQLiteMemory(tmp_path / "test.sqlite")
    return store


@pytest.fixture
def mock_chroma():
    """
    Fully mocked ChromaAmbient — no real embeddings, no Rust tokenizer.
    Returns an AsyncMock that satisfies the interface used in process_message().
    """
    chroma = MagicMock()
    chroma.add_message = AsyncMock(return_value=None)
    chroma.search = AsyncMock(return_value=[])
    chroma.recent_window = AsyncMock(return_value=[])
    return chroma


# ─────────────────────────────────────────────────────────────────────────────
# Settings fixture — inject safe test defaults
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def test_settings(tmp_path, monkeypatch):
    """
    Override env vars so the Settings object never hits real APIs or real paths.
    Applied to ALL tests automatically (autouse=True).
    """
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-not-real")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_ENABLED", "0")
    monkeypatch.setenv("LIVE_SEARCH_ENABLED", "1")
    monkeypatch.setenv("ALLOWED_CHAT_IDS", "")
    monkeypatch.setenv("ALLOW_ALL_CHATS", "1")
    monkeypatch.setenv("BOT_PERSONA_NAME", "Shimmi")
    monkeypatch.setenv("APP_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("FACTS_MIN_CONF", "0.7")
    monkeypatch.setenv("GEMINI_ENABLED", "0")
    monkeypatch.setenv("MISTRAL_API_KEY", "")  # disable Mistral in tests
    monkeypatch.setenv("GEMINI_API_KEY", "")   # disable Gemini in tests
    # Force provider chain to groq_8b-only so no Gemini/Mistral client needed
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key-not-real")


# ─────────────────────────────────────────────────────────────────────────────
# Provider lock — forces groq_8b for ALL tests (autouse)
# ─────────────────────────────────────────────────────────────────────────────
# Settings() is instantiated at module import time, so monkeypatch.setenv for
# GEMINI_API_KEY has no effect on already-loaded settings.gemini_api_key.
# Instead we patch _pick_provider_and_model directly so every run_agent()
# call gets groq_8b regardless of which LLM fixture the test uses.
# This prevents "Gemini client not initialised" in all integration tests.

@pytest.fixture(autouse=True)
def force_groq_provider():
    """Force all LLM calls to use groq_8b — no Gemini/Mistral clients needed."""
    with patch(
        "app.agent_engine._pick_provider_and_model",
        return_value=("groq_8b", "llama-3.1-8b-instant"),
    ):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# LLM mock fixture — makes _groq_raw() return canned responses
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_answer():
    """
    Mock _groq_raw to always return a canned 'answer' orchestrator result,
    plus empty extract/verify/format results.
    """
    call_count = [0]

    async def _fake_groq_raw(messages, *, max_tokens, chat_id, label, role, timeout=None):
        call_count[0] += 1
        if role == "orchestrate":
            return CANNED_ORCHESTRATOR_ANSWER
        elif label in ("extract", "extract_fb"):
            return CANNED_EMPTY_EXTRACT
        elif label in ("verify", "verify_fb"):
            return CANNED_VERIFY_RESULT
        elif label in ("format", "format_fb"):
            return CANNED_FORMAT_RESULT
        return CANNED_EMPTY_EXTRACT

    with patch("app.agent_engine._groq_raw", side_effect=_fake_groq_raw) as mock:
        mock.call_count_ref = call_count
        yield mock


@pytest.fixture
def mock_llm_search():
    """
    Mock _groq_raw to simulate search → answer flow.
    First orchestrator call returns action=search with weather tool_call.
    Second orchestrator call returns action=answer.
    """
    orch_calls = [0]

    async def _fake_groq_raw(messages, *, max_tokens, chat_id, label, role, timeout=None):
        if role == "orchestrate":
            orch_calls[0] += 1
            if orch_calls[0] == 1:
                return CANNED_ORCHESTRATOR_SEARCH
            return CANNED_ORCHESTRATOR_SEARCH_THEN_ANSWER
        elif label.startswith("extract"):
            return CANNED_EMPTY_EXTRACT
        elif label.startswith("verify"):
            return CANNED_VERIFY_RESULT
        elif label.startswith("format"):
            return CANNED_FORMAT_RESULT
        return CANNED_EMPTY_EXTRACT

    with patch("app.agent_engine._groq_raw", side_effect=_fake_groq_raw):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher mock — prevents real HTTP calls during integration tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_tool_dispatcher():
    """
    Replace ToolDispatcher.dispatch with a mock that returns canned data.
    Verifies the dispatcher was called with a correct tool, but never hits HTTP.
    """
    with patch("app.tools.tool_dispatcher.dispatch", new_callable=AsyncMock) as mock:
        mock.return_value = "🌤️ Hyderabad: 32°C, sunny. Humidity 65%."
        yield mock


# ─────────────────────────────────────────────────────────────────────────────
# Marks
# ─────────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires real LLM API keys (costs quota)")
    config.addinivalue_line("markers", "integration: requires running Shimmi bot")
