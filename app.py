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
from mcp_server.database import DEFAULT_DB, connect, reset_database

load_dotenv()
configure_logging()
st.set_page_config(page_title="IT Service Desk Agent", page_icon="🛠️", layout="wide")

EXAMPLES = [
    "Triage all open tickets.",
    "Investigate T-101, fix the root cause and resolve it.",
    "Users say GitHub is down. Is it us or them?",
    "Check the TLS certificate and DNS of openai.com.",
    "Handle T-104.",
]
LABELS = {"it_diagnostics": "🕵️ subagent", "internet_checker": "🌐 subagent", "load_skill": "📘 skill",
          "jev_triage": "⚡ Jev", "get_ticket": "🔎 MCP read", "list_tickets": "🔎 MCP read",
          "check_service": "🔎 MCP read", "run_sql": "🔎 MCP read",
          "restart_service": "⚠️ MCP write", "update_ticket": "⚠️ MCP write",
          "search_past_incidents": "🧠 memory read", "save_lesson": "🧠 memory write",
          "record_incident": "🧠 memory write"}


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
if not DEFAULT_DB.exists():
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
    jev = agent.jev
    if jev is not None and jev.disabled_reason:
        st.warning(f"Jev: {jev.disabled_reason} → using LLM fallback")
    elif os.getenv("TYPESAFE_API_KEY"):
        st.caption("Jev: TypeSafe System One ✅")
    else:
        st.caption("Jev: no TYPESAFE_API_KEY → LLM fallback")
    st.button("➕ New chat", on_click=new_chat, type="primary", use_container_width=True,
              help="Start a new conversation. Long-term memory (lessons, past incidents) carries over.")
    st.caption(f"Chat `{st.session_state.thread[:8]}`")
    if st.button("Reset demo data & chat", help="Long-term memory is kept, so you can see the agent reuse it."):
        reset_database()
        st.session_state.clear()
        st.rerun()
    with st.expander("Architecture"):
        st.markdown(
            "1. **Plan** – restate the request, list the steps\n"
            "2. **Execute** – direct lookups for simple questions; **Jev** for typed triage; "
            "subagents for multi-step work\n"
            "   - `it_diagnostics`: internal systems via **MCP** (read-only)\n"
            "   - `internet_checker`: **real** websites, DNS, TLS, vendor status\n"
            "3. **Approve** – restarts / ticket updates wait for you\n"
            "4. **Answer** – findings, actions, evidence\n"
            "5. **Remember** – approved lessons and resolved incidents, reused in later chats\n\n"
            "Skills: " + ", ".join(f"`{s}`" for s in load_all())
        )
    st.subheader("Environment (live)")
    with connect(DEFAULT_DB) as conn:
        st.markdown("**Services**")
        st.dataframe(pd.read_sql("SELECT name, status, memory_pct, error_rate_pct FROM services", conn),
                     hide_index=True)
        st.markdown("**Tickets**")
        st.dataframe(pd.read_sql("SELECT id, title, priority, status FROM tickets", conn), hide_index=True)
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
    cols = st.columns(len(EXAMPLES))
    clicks = [col.button(ex, use_container_width=True) for col, ex in zip(cols, EXAMPLES)]
    prompt = st.chat_input("Describe the problem...") or next((ex for ex, hit in zip(EXAMPLES, clicks) if hit), None)
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        steps = stream_turn(agent.stream_chat(st.session_state.thread, prompt))
        st.session_state.history.append(("assistant", steps))
        st.session_state.pending = pending(steps)
        st.rerun()
