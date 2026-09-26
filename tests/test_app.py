"""UI flow: plan shown first, tool steps shown, approval card, approve -> answer, reset (scripted LLM)."""

import asyncio
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from agent.agent import ServiceDeskAgent
from mcp_server import server
from tests.conftest import ScriptedLLM, call, delegate, say, verdict

APP = str(Path(__file__).resolve().parent.parent / "app.py")
PLAN = ("**Understanding:** You're asking me to run the checkout, payments and login failures as a major incident."
        "\n\n**Plan:** To do this, I need to:\n1. Ask it_diagnostics to investigate\n"
        "2. Fix the root cause (approval)")


def open_app(agent: ServiceDeskAgent, monkeypatch: pytest.MonkeyPatch, thread: str = "ui") -> AppTest:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["agent"] = agent
    at.session_state["thread"] = thread
    at.session_state["history"] = []
    at.session_state["pending"] = []
    at.run()
    return at


def change_status(change_id: str) -> str:
    return server.run_sql(f"SELECT status FROM changes WHERE id = '{change_id}'")[0]["status"]


def test_plan_then_execution_then_approval(incident_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(scripts={
        "planner": [say(PLAN)],
        "main": [delegate("it_diagnostics", "Investigate checkout, payments and logins"),
                 call("rollback_change", change_id="CHG-231", reason="connection storm against core-db"),
                 say("Rolled back CHG-231; core-db and its clients are healthy again.")],
        "it_diagnostics": [call("check_service", service="core-db"), say("core-db pool exhausted by payment-api")],
    })
    at = open_app(ServiceDeskAgent(llm, incident_db), monkeypatch)
    next(b for b in at.button if b.label.startswith("Checkout, payments and logins")).click().run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "run the checkout, payments and login failures as a major incident" in text  # plan first
    assert "it_diagnostics" in text and "check_service" in text  # execution, incl. subagent steps
    assert any("Approval required" in m.value for m in at.markdown)
    assert change_status("CHG-231") == "applied"

    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception
    assert change_status("CHG-231") == "rolled_back" and server.check_service("core-db")["status"] == "healthy"
    assert "Rolled back CHG-231" in " ".join(m.value for m in at.markdown)


def test_reset_demo_undoes_everything(incident_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("plan")], "main": [say("First answer.")]})
    agent = ServiceDeskAgent(llm, incident_db)
    at = open_app(agent, monkeypatch, thread="before")
    at.chat_input[0].set_value("How is core-db?").run()
    server.rollback_change("CHG-231", "fixed in an earlier run")
    server.update_ticket("T-101", "resolved", "fixed")
    asyncio.run(agent.memory.add_lesson("Page data-team for core-db problems", "they own it"))
    asyncio.run(agent.memory.add_incident("payment-api", "T-101", "checkout HTTP 500", "CHG-231 connection storm",
                                          "roll back CHG-231", "core-db 100/100 -> healthy"))

    next(b for b in at.button if "Reset demo" in b.label).click().run()
    assert not at.exception
    assert change_status("CHG-231") == "applied" and server.check_service("core-db")["status"] == "degraded"
    assert server.get_ticket("T-101")["status"] == "open"
    assert server.run_sql("SELECT COUNT(*) n FROM audit_log")[0]["n"] == 0
    assert agent.memory.lessons() == "" and agent.memory.incidents() == []  # nothing carries over to skip work
    assert at.session_state["thread"] != "before" and at.session_state["history"] == []
    assert "First answer." not in " ".join(m.value for m in at.markdown)


def test_new_chat_starts_fresh_but_keeps_long_term_memory(incident_db: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("plan 1"), say("plan 2")],
                               "main": [say("First answer."), say("Second answer.")]})
    agent = ServiceDeskAgent(llm, incident_db)
    asyncio.run(agent.memory.add_lesson("Page app-team for cache problems", "they own it"))
    at = open_app(agent, monkeypatch, thread="first")
    at.chat_input[0].set_value("How is the cache?").run()
    assert "First answer." in " ".join(m.value for m in at.markdown)

    next(b for b in at.button if "New chat" in b.label).click().run()
    assert not at.exception
    assert at.session_state["thread"] != "first" and at.session_state["history"] == []
    assert "First answer." not in " ".join(m.value for m in at.markdown)  # the chat is fresh
    assert "Page app-team" in " ".join(m.value for m in at.markdown)  # sidebar still shows long-term memory
    at.chat_input[0].set_value("And now?").run()
    assert "Page app-team" in llm.prompts("main")[-1]  # and still used in the new chat


def test_critic_note_and_failed_verification_are_shown(incident_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    llm = ScriptedLLM(scripts={
        "planner": [say("plan")],
        "main": [call("restart_service", service="core-db", reason="connections exhausted"),
                 say("The restart did not hold; next I will look at recent changes.")],
        "critic": [verdict("weak", "core-db is exhausted, but nobody checked who opens the connections")],
    })
    at = open_app(ServiceDeskAgent(llm, incident_db), monkeypatch)
    at.chat_input[0].set_value("Fix the outage").run()
    assert any("nobody checked who opens the connections" in w.value for w in at.warning)  # critic on the card

    next(b for b in at.button if "Approve" in b.label).click().run()
    assert not at.exception
    assert any("FAILED" in e.value and "relapsed:" in e.value and "core-db" in e.value for e in at.error)
