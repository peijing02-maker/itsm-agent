"""UI flow: plan shown first, tool steps shown, approval card, approve -> answer (scripted LLM)."""

import asyncio
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.agent import ServiceDeskAgent
from mcp_server import server
from tests.conftest import ScriptedLLM, call, say

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PLAN = ("**Understanding:** You're asking about the failing web shop (T-101).\n\n"
        "**Plan:** To do this, I need to:\n1. Ask it_diagnostics to investigate\n2. Restart the root cause (approval)")


def test_plan_then_execution_then_approval(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = ScriptedLLM(scripts={
        "planner": [say(PLAN)],
        "main": [call("it_diagnostics", task="Investigate T-101"),
                 call("restart_service", service="cache", reason="memory 97%"),
                 say("Cache restarted; web-shop is healthy again.")],
        "it_diagnostics": [call("check_service", service="cache"), say("cache memory 97% is the root cause")],
    })
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["agent"] = ServiceDeskAgent(llm, db)
    at.session_state["thread"] = "ui"
    at.session_state["history"] = []
    at.session_state["pending"] = []
    at.run()
    next(b for b in at.button if b.label.startswith("Investigate T-101")).click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "You're asking about the failing web shop" in text  # plan first
    assert "it_diagnostics" in text and "check_service" in text  # execution, incl. subagent steps
    assert any("Approval required" in m.value for m in at.markdown)
    assert server.check_service("cache")["status"] == "degraded"

    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception
    assert server.check_service("cache")["status"] == "healthy"
    assert "Cache restarted" in " ".join(m.value for m in at.markdown)


def test_new_chat_starts_fresh_but_keeps_long_term_memory(db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = ScriptedLLM(scripts={"planner": [say("plan 1"), say("plan 2")],
                               "main": [say("First answer."), say("Second answer.")]})
    agent = ServiceDeskAgent(llm, db)
    asyncio.run(agent.memory.add_lesson("Page app-team for cache problems", "they own it"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["agent"] = agent
    at.session_state["thread"] = "first"
    at.session_state["history"] = []
    at.session_state["pending"] = []
    at.run()
    at.chat_input[0].set_value("How is the cache?").run()
    assert "First answer." in " ".join(m.value for m in at.markdown)

    next(b for b in at.button if "New chat" in b.label).click().run()
    assert not at.exception
    assert at.session_state["thread"] != "first" and at.session_state["history"] == []
    assert "First answer." not in " ".join(m.value for m in at.markdown)  # the chat is fresh
    assert "Page app-team" in " ".join(m.value for m in at.markdown)  # sidebar still shows long-term memory
    at.chat_input[0].set_value("And now?").run()
    assert "Page app-team" in llm.prompts("main")[-1]  # and still used in the new chat
