"""Level 2 - the real LangChain agent (graph, MCP, subagents, skills, approvals) driven by a scripted LLM."""

from pathlib import Path

from agent.agent import ServiceDeskAgent
from agent.control import VERIFY_MINUTES
from mcp_server import server
from tests.conftest import ScriptedLLM, call, delegate, say, together, verdict

PLAN_3 = ("**Understanding:** You're asking about the failing web shop.\n\n**Plan:** To do this, I need to:\n"
          "1. Investigate with it_diagnostics\n2. Fix the root cause (approval)\n3. Resolve the ticket (approval)")


def kinds(result) -> list[tuple[str, str, str]]:
    return [(s.kind, s.name, s.by) for s in result.steps if s.kind != "plan"]


async def state(agent: ServiceDeskAgent, thread: str) -> dict:
    return (await agent.graph.aget_state({"configurable": {"thread_id": thread}})).values


# ------------------------------------------------------------------ delegation
async def test_delegates_investigation_to_subagent(db: Path) -> None:
    llm = ScriptedLLM(scripts={
        "main": [delegate("it_diagnostics", "Why is web-shop failing?"), say("Root cause: cache memory at 97%.")],
        "it_diagnostics": [call("check_service", service="web-shop"), call("check_service", service="cache"),
                           say("web-shop depends on cache; cache memory 97% -> root cause.")],
    })
    result = await ServiceDeskAgent(llm, db).achat("t1", "Investigate T-101")
    assert kinds(result) == [
        ("tool_call", "it_diagnostics", "agent"),  # shown as the specialist, not as the `task` plumbing
        ("tool_call", "check_service", "it_diagnostics"),  # subagent steps are streamed live too
        ("tool_result", "check_service", "it_diagnostics"),
        ("tool_call", "check_service", "it_diagnostics"),
        ("tool_result", "check_service", "it_diagnostics"),
        ("tool_result", "it_diagnostics", "agent"),
        ("answer", "", "agent"),
    ]
    assert "root cause" in result.steps[-2].content  # only the report reaches the main agent


async def test_parallel_subagents_are_attributed_exactly(incident_db: Path) -> None:
    llm = ScriptedLLM(scripts={
        "main": [together(delegate("it_diagnostics", "Follow core-db dependencies"),
                          delegate("change_analyst", "What changed before the errors?")),
                 say("core-db is exhausted by payment-api after CHG-231.")],
        "it_diagnostics": [call("check_service", service="core-db"), say("core-db pool exhausted")],
        "change_analyst": [call("list_changes"), say("CHG-231 landed 2 minutes before the symptoms")],
    })
    result = await ServiceDeskAgent(llm, incident_db).achat("t", "Why is checkout failing?")
    sub_steps = {(s.name, s.by) for s in result.steps if s.kind in ("tool_call", "tool_result") and s.by != "agent"}
    assert sub_steps == {("check_service", "it_diagnostics"), ("list_changes", "change_analyst")}
    reports = {s.name: s.content for s in result.steps if s.kind == "tool_result" and s.by == "agent"}
    assert "exhausted" in reports["it_diagnostics"] and "CHG-231" in reports["change_analyst"]


async def test_runaway_subagent_is_capped_in_code(db: Path) -> None:
    from agent.subagents import SUBAGENT_TOOL_CALL_LIMIT

    llm = ScriptedLLM(scripts={
        "main": [delegate("it_diagnostics", "look at everything"), say("done")],
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
    assert not [s for s in result.steps if s.kind == "todos"]  # a 1-step request gets no todo list


# ---------------------------------------------------------- planning as state
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
    assert plan in [m.text for m in (await state(agent, "t4"))["messages"]]  # the executor sees its plan


async def test_multi_step_plan_becomes_a_todo_list_the_agent_updates(db: Path) -> None:
    done_first = [{"content": "Investigate with it_diagnostics", "status": "completed"},
                  {"content": "Page app-team instead of restarting", "status": "in_progress"}]
    llm = ScriptedLLM(scripts={"planner": [say(PLAN_3)],
                               "main": [call("write_todos", todos=done_first), say("Paged.")]})
    agent = ServiceDeskAgent(llm, db)
    result = await agent.achat("t", "Fix the web shop")
    todos = [s.content for s in result.steps if s.kind == "todos"]
    assert [t["status"] for t in todos[0]] == ["in_progress", "pending", "pending"]  # seeded from the plan
    assert todos[-1] == done_first  # the agent replanned, and the UI saw it
    assert (await state(agent, "t"))["todos"] == done_first
    assert not [s for s in result.steps if s.kind == "tool_result" and s.name == "write_todos"]  # no noise


async def test_memory_keeps_context_within_a_thread(db: Path) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("plan 1"), say("plan 2")],
                               "main": [say("Noted."), say("You asked about T-102.")]})
    agent = ServiceDeskAgent(llm, db)
    await agent.achat("t5", "Remember ticket T-102")
    await agent.achat("t5", "Which ticket did I mention?")
    humans = [m.text for m in (await state(agent, "t5"))["messages"] if m.type == "human"]
    assert humans == ["Remember ticket T-102", "Which ticket did I mention?"]


def test_sync_stream_for_streamlit_yields_steps_in_order(db: Path) -> None:
    llm = ScriptedLLM(scripts={"planner": [say("**Understanding:** ...")],
                               "main": [call("current_time"), say("It is late.")]})
    kinds_seen = [s.kind for s in ServiceDeskAgent(llm, db).stream_chat("t7", "What time is it?")]
    assert kinds_seen == ["plan_token", "plan", "tool_call", "tool_result", "answer"]


# ------------------------------------------------------- approval & guardrails
async def test_write_action_waits_for_human_approval_then_is_verified(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("restart_service", service="cache", reason="memory 97%"),
                                        say("Cache restarted and web-shop is healthy.")]})
    agent = ServiceDeskAgent(llm, db)
    result = await agent.achat("t2", "Fix the web shop")
    assert result.pending and result.pending[0]["name"] == "restart_service"
    assert server.check_service("cache")["status"] == "degraded"  # nothing happened yet

    result = await agent.aresume("t2", approved=True)
    assert [s.kind for s in result.steps] == ["tool_result", "verification", "answer"]  # call not shown twice
    check = result.steps[1].content
    assert check["passed"] and check["watched"] == ("cache", "payment-api", "web-shop")  # all services connected to the fix
    assert f"VERIFICATION ({VERIFY_MINUTES} min observed, in code): PASSED" in result.steps[0].content
    assert server.check_service("web-shop")["status"] == "healthy"


async def test_rejected_action_never_runs(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("update_ticket", ticket_id="T-102", status="resolved", note="done"),
                                        say("OK, I left T-102 open.")]})
    agent = ServiceDeskAgent(llm, db)
    await agent.achat("t3", "Close T-102")
    result = await agent.aresume("t3", approved=False, reason="Not verified yet")
    assert server.get_ticket("T-102")["status"] == "open"
    assert any("Not verified yet" in str(s.content) for s in result.steps if s.kind == "tool_result")


async def test_prompt_injection_cannot_bypass_approval(db: Path) -> None:
    # Even if injected ticket text fooled the LLM, closing a ticket still needs a human.
    llm = ScriptedLLM(scripts={"main": [call("update_ticket", ticket_id="T-102", status="resolved", note="injected")]})
    result = await ServiceDeskAgent(llm, db).achat("t6", "Handle T-104")
    assert result.pending and server.get_ticket("T-102")["status"] == "open"


async def test_cannot_resolve_a_ticket_while_its_service_is_unhealthy(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("update_ticket", ticket_id="T-101", status="resolved", note="fixed"),
                                        say("T-101 stays open until web-shop is healthy.")]})
    result = await ServiceDeskAgent(llm, db).achat("t", "Close T-101")
    assert not result.pending  # blocked in code before a human is asked
    blocked = next(s.content for s in result.steps if s.kind == "tool_result" and s.name == "update_ticket")
    assert "its service web-shop is degraded" in blocked
    assert server.get_ticket("T-101")["status"] == "open"


async def test_critic_verdict_is_shown_on_the_approval_card(db: Path) -> None:
    llm = ScriptedLLM(scripts={
        "main": [call("restart_service", service="cache", reason="memory 97%")],
        "critic": [verdict("supported", "cache memory is 97% and web-shop depends on it")],
    })
    result = await ServiceDeskAgent(llm, db).achat("t", "Fix the web shop")
    assert result.pending[0]["note"] == "Change critic: supported: cache memory is 97% and web-shop depends on it"


async def test_critic_failure_leaves_the_decision_to_the_human(db: Path) -> None:
    llm = ScriptedLLM(scripts={"main": [call("restart_service", service="cache", reason="memory 97%")],
                               "critic": [say("not a structured answer")]})
    result = await ServiceDeskAgent(llm, db).achat("t", "Fix the web shop")
    assert result.pending and "unavailable" in result.pending[0]["note"]


async def test_critic_blocks_a_contradicted_fix_before_a_human_is_asked(db: Path) -> None:
    llm = ScriptedLLM(scripts={
        "main": [call("restart_service", service="web-shop", reason="HTTP 500"), say("I will look deeper.")],
        "critic": [verdict("contradicted", "web-shop is a symptom: its dependency cache is at 97% memory")],
    })
    agent = ServiceDeskAgent(llm, db)
    result = await agent.achat("t", "Fix the web shop")
    assert not result.pending
    blocked = next(s.content for s in result.steps if s.kind == "tool_result" and s.name == "restart_service")
    assert blocked.startswith("Blocked by the change critic") and "symptom" in blocked
    assert server.run_sql("SELECT COUNT(*) n FROM audit_log")[0]["n"] == 0  # nothing ran
    assert (await state(agent, "t"))["needs_replan"] is True


async def test_failed_verification_forces_a_revised_plan_before_the_next_fix(incident_db: Path) -> None:
    replan = [{"content": "Restart only masked it: roll back CHG-231 (payment-api deploy)", "status": "in_progress"}]
    llm = ScriptedLLM(scripts={
        "main": [call("restart_service", service="core-db", reason="connections exhausted"),
                 call("rollback_change", change_id="CHG-231", reason="retry storm"),  # blocked: plan not revised
                 call("write_todos", todos=replan),
                 call("rollback_change", change_id="CHG-231", reason="retry storm"),
                 say("Rolled back CHG-231; all services healthy.")],
    })
    agent = ServiceDeskAgent(llm, incident_db)
    result = await agent.achat("t", "Fix the checkout outage")
    assert [p["name"] for p in result.pending] == ["restart_service"]

    result = await agent.aresume("t", approved=True)
    first = next(s.content for s in result.steps if s.kind == "verification")
    assert not first["passed"] and "core-db" in first["relapsed"]  # the restart only masked the cause
    gate = next(s.content for s in result.steps if s.kind == "tool_result" and s.name == "rollback_change")
    assert gate.startswith("Blocked before approval: your last fix FAILED verification")
    assert [s.content for s in result.steps if s.kind == "todos"] == [replan]
    assert [p["name"] for p in result.pending] == ["rollback_change"]  # allowed again after replanning

    result = await agent.aresume("t", approved=True)
    assert next(s.content for s in result.steps if s.kind == "verification")["passed"]
    assert {r["status"] for r in server.run_sql("SELECT status FROM services")} == {"healthy"}
    assert server.run_sql("SELECT status FROM changes WHERE id = 'CHG-231'")[0]["status"] == "rolled_back"
