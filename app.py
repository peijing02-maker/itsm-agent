"""Streamlit demo: chat with the IT Service Desk agent and watch it plan and work live.

Run:  streamlit run app.py
"""

import json
import os
import uuid
from collections.abc import Iterator

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent.agent import ServiceDeskAgent, Step
from agent.logging_config import configure_logging
from agent.skills import load_all
from mcp_server.database import DEFAULT_DB, connect, current_scenario, reset_database
from mcp_server.scenarios import SCENARIOS

load_dotenv()
configure_logging()
st.set_page_config(page_title="IT Service Desk Agent", page_icon="🛠️", layout="wide")

EXAMPLES = {
    "cache-outage": [
        "Triage all open tickets.",
        "Investigate T-101, fix the root cause and resolve it.",
        "Users say GitHub is down. Is it us or them?",
        "Check the TLS certificate and DNS of openai.com.",
        "Handle T-104.",
    ],
    "major-incident": [
        "Triage all open tickets.",
        ("Checkout, payments and logins are failing. Run this as a major incident: find the root cause, fix it "
         "and resolve the related tickets."),
        "What changed in the last hour, and could it explain the errors?",
        "Is Cloudflare having an outage, or is it us?",
    ],
}
LABELS = {"it_diagnostics": "🕵️ subagent", "change_analyst": "🔀 subagent", "internet_checker": "🌐 subagent",
          "load_skill": "📘 skill", "triage_tickets": "⚡ triage",
          "get_ticket": "🔎 MCP read", "list_tickets": "🔎 MCP read", "check_service": "🔎 MCP read",
          "run_sql": "🔎 MCP read", "list_changes": "🔎 MCP read", "get_metrics": "🔎 MCP read",
          "get_logs": "🔎 MCP read", "search_knowledge_base": "🔎 MCP read",
          "restart_service": "⚠️ fix", "flush_cache": "⚠️ fix", "rollback_change": "⚠️ fix",
          "update_ticket": "✏️ MCP write", "create_ticket": "✏️ MCP write", "link_tickets": "✏️ MCP write",
          "page_team": "📟 page", "write_todos": "📋 plan", "ls": "🗒️ notes", "read_file": "🗒️ notes",
          "write_file": "🗒️ notes", "edit_file": "🗒️ notes",
          "search_past_incidents": "🧠 memory read", "save_lesson": "🧠 memory write",
          "record_incident": "🧠 memory write"}
TODO_ICONS = {"completed": "✅", "in_progress": "▶️", "pending": "⬜"}


# ---------------------------------------------------------------- rendering
def render_plan(plan: str) -> None:
    with st.container(border=True):
        st.markdown("🧠 " + plan)


def render_step(step: Step) -> None:
    """One execution step. Subagent steps are indented under the subagent that made them."""
    indent = "&nbsp;" * 8 + "↳ " if step.by != "agent" else ""
    who = f"*{step.by}* " if step.by != "agent" else ""
    if step.kind == "tool_call":
        label = LABELS.get(step.name, "🔧 tool")
        st.markdown(f"{indent}{who}{label} → **{step.name}** `{json.dumps(step.content)[:160]}`")
    elif step.kind == "tool_result":
        with st.expander(f"{'↳ ' if who else ''}✓ result of {step.name}"):
            st.text(str(step.content)[:4000])
    elif step.kind == "todos":
        st.markdown("📋 **Plan checklist**\n" + "\n".join(
            f"- {TODO_ICONS.get(t['status'], '⬜')} {t['content']}" for t in step.content))
    elif step.kind == "verification":
        v = step.content
        text = f"**Verification** of {v['action']} ({v['minutes']} min, watched {', '.join(v['watched'])})"
        if v["passed"]:
            st.success(f"✅ {text}: PASSED")
        else:
            relapsed = f"; relapsed: {', '.join(v['relapsed'])}" if v["relapsed"] else ""
            st.error(f"❌ {text}: FAILED ({', '.join(f'{k} {s}' for k, s in v['unhealthy'].items())}{relapsed}). "
                     "The agent must revise its plan.")
    elif step.kind == "approval_needed":
        st.warning(f"✋ Waiting for your approval: **{step.name}** `{json.dumps(step.content)}`")
    elif step.kind == "answer":
        st.markdown("---\n" + step.content)


def render_turn(steps: list[Step]) -> None:
    for s in steps:
        render_plan(s.content) if s.kind == "plan" else render_step(s)


def stream_turn(events: Iterator[Step]) -> list[Step]:
    """Render live: the plan streams token by token, then each tool call appears as it happens."""
    steps: list[Step] = []
    with st.chat_message("assistant"):
        plan_box = st.empty()
        status = st.status("Working...", expanded=True)
        plan = ""
        for step in events:
            if step.kind == "plan_token":
                plan += step.content
                plan_box.container(border=True).markdown("🧠 " + plan + " ▌")
                continue
            if step.kind == "plan":
                plan_box.container(border=True).markdown("🧠 " + step.content)
                status.update(label="Executing the plan...")
            else:
                with status:
                    render_step(step)
            steps.append(step)
        needs_approval = any(s.kind == "approval_needed" for s in steps)
        status.update(label="Paused: waiting for approval" if needs_approval else "Done",
                      state="running" if needs_approval else "complete")
    return steps


def pending(steps: list[Step]) -> list[Step]:
    return [s for s in steps if s.kind == "approval_needed"]


# -------------------------------------------------------------------- state
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Copy `.env.example` to `.env`, add your key, and restart.")
    st.stop()
if current_scenario() is None:  # missing, or created by an older version without scenarios
    reset_database()
def new_chat() -> None:
    """Start a new conversation: fresh short-term memory (thread), same agent and long-term memory."""
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.history = []  # list of (role, str | list[Step])
    st.session_state.pending = []


if "agent" not in st.session_state:
    st.session_state.agent = ServiceDeskAgent()
    new_chat()
agent: ServiceDeskAgent = st.session_state.agent

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🛠️ Service Desk Agent")
    st.caption(f"LangChain · plan-and-execute · model `{os.getenv('OPENAI_MODEL', 'gpt-5.5')}`")
    st.button("➕ New chat", on_click=new_chat, type="primary", use_container_width=True,
              help="Start a new conversation. Long-term memory (lessons, past incidents) carries over.")
    st.caption(f"Chat `{st.session_state.thread[:8]}`")
    scenario = current_scenario() or "cache-outage"
    names = sorted(SCENARIOS)
    chosen = st.selectbox("Scenario", names, index=names.index(scenario),
                          help="\n\n".join(f"**{n}**: {SCENARIOS[n].description}" for n in names))
    if st.button("Reset demo data & chat", help="Long-term memory is kept, so you can see the agent reuse it."):
        reset_database(scenario=chosen)
        st.session_state.clear()
        st.rerun()
    st.caption(f"Loaded: `{scenario}`" + (" (reset to switch)" if chosen != scenario else ""))
    with st.expander("Architecture"):
        st.markdown(
            "1. **Plan** – restate the request, list the steps; multi-step plans become a live checklist\n"
            "2. **Execute** – direct lookups for simple questions; typed LLM triage; "
            "subagents (in parallel) for multi-step work\n"
            "   - `it_diagnostics`: internal systems via **MCP** (read-only)\n"
            "   - `change_analyst`: what changed vs. when symptoms began\n"
            "   - `internet_checker`: **real** websites, DNS, TLS, vendor status\n"
            "3. **Gate** – code gates, then a **change critic** reviews every fix\n"
            "4. **Approve** – fixes / ticket updates / pages wait for you\n"
            "5. **Verify** – code watches the service and its dependents; a failed fix forces a new plan\n"
            "6. **Remember** – approved lessons and resolved incidents, reused in later chats\n\n"
            "Skills: " + ", ".join(f"`{s}`" for s in load_all())
        )
    st.subheader("Environment (live)")
    with connect(DEFAULT_DB) as conn:
        st.markdown("**Services**")
        st.dataframe(pd.read_sql("SELECT name, status, cpu_pct, memory_pct, error_rate_pct, owner FROM services",
                                 conn), hide_index=True)
        st.markdown("**Tickets**")
        st.dataframe(pd.read_sql("SELECT id, title, priority, status, parent_id FROM tickets", conn),
                     hide_index=True)
        st.markdown("**Recent changes**")
        st.dataframe(pd.read_sql("SELECT id, ts, service, summary, status FROM changes ORDER BY ts DESC", conn),
                     hide_index=True)
        pages = pd.read_sql("SELECT ts, team, severity, message FROM pages ORDER BY id DESC", conn)
        if len(pages):
            st.markdown("**Pages**")
            st.dataframe(pages, hide_index=True)
        st.markdown("**Audit log**")
        st.dataframe(pd.read_sql("SELECT ts, action, target, detail FROM audit_log ORDER BY id DESC", conn),
                     hide_index=True)
    st.subheader("Agent memory (long-term)")
    lessons, incidents = agent.memory.lessons(), agent.memory.incidents()
    with st.expander(f"Lessons ({lessons.count(chr(10) + '- ')})"):
        st.markdown(lessons or "_None yet. Reject an action with a reason, or ask the agent to remember a rule._")
    with st.expander(f"Past incidents ({len(incidents)})"):
        st.markdown("\n\n---\n\n".join(incidents) or "_None yet. Resolve a verified fix to record one._")
    if st.button("Clear agent memory"):
        agent.memory.clear()
        st.rerun()

# --------------------------------------------------------------------- chat
st.title("IT Service Desk Agent")
st.caption("Tell me the problem. I'll say how I understand it and my plan, then you'll see every step I take.")

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content) if role == "user" else render_turn(content)

if st.session_state.pending:
    with st.container(border=True):
        st.markdown("### ✋ Approval required")
        several = len(st.session_state.pending) > 1
        keep = []
        for i, action in enumerate(st.session_state.pending):
            st.markdown(f"**{action.name}**")
            st.json(action.content)
            if action.note:
                (st.warning if ": weak:" in action.note else st.info)("🧐 " + action.note)
            # With several actions, each can be decided separately (e.g. resolve the ticket, skip the memory).
            key = f"approve-{st.session_state.thread}-{len(st.session_state.history)}-{i}"  # fresh per round
            keep.append(st.checkbox(f"Approve {action.name}", value=True, key=key) if several else True)
        reason = st.text_input("Reason (sent to the agent if you reject)")
        approve, reject = st.columns(2)
        decision = None
        approve_label = "✅ Approve selected" if several else "✅ Approve"
        if approve.button(approve_label, type="primary", use_container_width=True):
            decision = keep
        if reject.button("❌ Reject all" if several else "❌ Reject", use_container_width=True):
            decision = False
        if decision is not None:
            steps = stream_turn(agent.stream_resume(st.session_state.thread, approved=decision, reason=reason))
            st.session_state.history.append(("assistant", steps))
            st.session_state.pending = pending(steps)
            st.rerun()
else:
    examples = EXAMPLES.get(current_scenario() or "", [])
    cols = st.columns(len(examples)) if examples else []
    clicks = [col.button(ex, use_container_width=True) for col, ex in zip(cols, examples)]
    prompt = st.chat_input("Describe the problem...") or next((ex for ex, hit in zip(examples, clicks) if hit), None)
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        steps = stream_turn(agent.stream_chat(st.session_state.thread, prompt))
        st.session_state.history.append(("assistant", steps))
        st.session_state.pending = pending(steps)
        st.rerun()
