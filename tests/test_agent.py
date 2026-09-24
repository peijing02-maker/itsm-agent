"""Level 2 - the real LangChain agent (graph, MCP, subagents, skills, approvals) driven by a scripted LLM."""

from pathlib import Path

from agent.agent import ServiceDeskAgent
from mcp_server import server
from tests.conftest import ScriptedLLM, call, say


def kinds(result) -> list[tuple[str, str, str]]:
    return [(s.kind, s.name, s.by) for s in result.steps if s.kind != "plan"]


async def test_delegates_investigation_to_subagent(db: Path) -> None:
    llm = ScriptedLLM(scripts={
        "main": [call("it_diagnostics", task="Why is web-shop failing?"), say("Root cause: cache memory at 97%.")],
        "it_diagnostics": [call("check_service", service="web-shop"), call("check_service", service="cache"),
                           say("web-shop depends on cache; cache memory 97% -> root cause.")],
    })
    result = await ServiceDeskAgent(llm, db).achat("t1", "Investigate T-101")
    assert kinds(result) == [
        ("tool_call", "it_diagnostics", "agent"),
        ("tool_call", "check_service", "it_diagnostics"),  # subagent steps are streamed live too
        ("tool_result", "check_service", "it_diagnostics"),
        ("tool_call", "check_service", "it_diagnostics"),
        ("tool_result", "check_service", "it_diagnostics"),
        ("tool_result", "it_diagnostics", "agent"),
        ("answer", "", "agent"),
    ]
    report = result.steps[-2].content
    assert "root cause" in report and "check_service, check_service" in report


async def test_write_action_waits_for_human_approval(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("restart_service", service="cache", reason="memory 97%"),
                                        say("Cache restarted and web-shop is healthy.")]})
    agent = ServiceDeskAgent(llm, db)
    result = await agent.achat("t2", "Fix the web shop")
    assert result.pending and result.pending[0]["name"] == "restart_service"
    assert server.check_service("cache")["status"] == "degraded"  # nothing happened yet

    result = await agent.aresume("t2", approved=True)
    assert not result.pending and result.steps[-1].kind == "answer"
    assert [s.kind for s in result.steps] == ["tool_result", "answer"]  # approved call not shown twice
    assert server.check_service("cache")["status"] == "healthy"
    assert server.check_service("web-shop")["status"] == "healthy"


async def test_rejected_action_never_runs(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("update_ticket", ticket_id="T-101", status="resolved", note="done"),
                                        say("OK, I left T-101 open.")]})
    agent = ServiceDeskAgent(llm, db)
    await agent.achat("t3", "Close T-101")
    result = await agent.aresume("t3", approved=False, reason="Not verified yet")
    assert server.get_ticket("T-101")["status"] == "open"
    assert any("Not verified yet" in str(s.content) for s in result.steps if s.kind == "tool_result")


async def test_plan_is_streamed_first_then_executed(db: Path) -> None:
    plan = "**Understanding:** You're asking about triage.\n\n**Plan:** To do this, I need to:\n1. Load incident-triage"
    llm = ScriptedLLM(scripts={
        "planner": [say(plan)],
        "main": [call("load_skill", name="incident-triage"), say("T-101 is high priority (App team).")],
    })
    agent = ServiceDeskAgent(llm, db)
    steps = [s async for s in agent.astream_chat("t4", "Triage the open tickets")]
    assert steps[0].kind == "plan_token" and steps[1].kind == "plan"  # plan shown before any tool runs
    assert steps[1].content.startswith("**Understanding:** You're asking about")
    skill_text = next(s.content for s in steps if s.kind == "tool_result" and s.name == "load_skill")
    assert "impact x urgency" in skill_text  # skill loaded on demand
    state = await agent.graph.aget_state({"configurable": {"thread_id": "t4"}})
    assert plan in [m.text for m in state.values["messages"]]  # the executor sees its plan


async def test_memory_keeps_context_within_a_thread(db: Path) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("plan 1"), say("plan 2")],
                               "main": [say("Noted."), say("You asked about T-102.")]})
    agent = ServiceDeskAgent(llm, db)
    await agent.achat("t5", "Remember ticket T-102")
    await agent.achat("t5", "Which ticket did I mention?")
    state = await agent.graph.aget_state({"configurable": {"thread_id": "t5"}})
    humans = [m.text for m in state.values["messages"] if m.type == "human"]
    assert humans == ["Remember ticket T-102", "Which ticket did I mention?"]


async def test_prompt_injection_cannot_bypass_approval(db: Path) -> None:
    # Even if injected ticket text fooled the LLM, closing a ticket still needs a human.
    llm = ScriptedLLM(scripts={"main": [call("update_ticket", ticket_id="T-102", status="resolved", note="injected")]})
    result = await ServiceDeskAgent(llm, db).achat("t6", "Handle T-104")
    assert result.pending and server.get_ticket("T-102")["status"] == "open"


def test_sync_stream_for_streamlit_yields_steps_in_order(db: Path) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("**Understanding:** ...")],
                               "main": [call("current_time"), say("It is late.")]})
    kinds_seen = [s.kind for s in ServiceDeskAgent(llm, db).stream_chat("t7", "What time is it?")]
    assert kinds_seen == ["plan_token", "plan", "tool_call", "tool_result", "answer"]


async def test_runaway_subagent_is_capped_in_code(db: Path) -> None:
    from agent.subagents import SUBAGENT_TOOL_CALL_LIMIT

    llm = ScriptedLLM(scripts={
        "main": [call("it_diagnostics", task="look at everything"), say("done")],
        "it_diagnostics": [call("check_service", service="cache") for _ in range(15)] + [say("report")],
    })
    result = await ServiceDeskAgent(llm, db).achat("t8", "Investigate")
    executed = [s for s in result.steps if s.kind == "tool_result" and s.by == "it_diagnostics"
                and '"memory_pct"' in str(s.content)]
    assert len(executed) == SUBAGENT_TOOL_CALL_LIMIT


async def test_simple_question_uses_one_direct_tool(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("check_service", service="cache"), say("Cache is degraded (97% memory).")]})
    result = await ServiceDeskAgent(llm, db).achat("t9", "What's the status of the cache?")
    assert [(s.name, s.by) for s in result.steps if s.kind == "tool_call"] == [("check_service", "agent")]
