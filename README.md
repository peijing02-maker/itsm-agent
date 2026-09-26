# IT Service Desk Agent

An **agentic AI** system built with **LangChain**. An IT service-desk agent works tickets end to end:
- as soon as you ask, it replies with how it understood the request and its plan, then shows every step live;
- it answers simple questions with one direct lookup, and triages tickets with a typed LLM classifier (structured output, confidence per decision);
- it tracks multi-step work as a **live plan checklist** it revises when evidence changes;
- it delegates investigation to **subagents**, in parallel when useful (internal diagnostics, change analysis, internet);
- it reads internal systems through an **MCP server**;
- it checks the **real internet** (websites, DNS, TLS, vendor status pages);
- it follows team procedures stored as **skills**;
- a **change critic** reviews every fix, and it fixes problems only **after a human approves**;
- it **verifies** every fix in code; a fix that doesn't hold forces a revised plan;
- it **remembers** across conversations: lessons from human feedback and resolved incidents (a human approves
  every memory write).

```
app.py                     Streamlit UI: chat, plan checklist, agent steps, approval card, live environment
agent/agent.py             main agent (LangChain create_agent + deepagents middleware), streaming to the UI
agent/planning.py          plan as state: planner steps seed the todo list (write_todos)
agent/subagents.py         it_diagnostics + change_analyst + internet_checker behind deepagents' `task` tool
agent/control.py           action gates (before approval) + verification of every fix (after it runs)
agent/critic.py            change critic: structured second opinion on every fix before a human sees it
agent/memory.py            long-term memory on deepagents: lessons in every prompt, past incidents on demand
agent/real_tools.py        real internet tools: website check, DNS, TLS expiry, vendor status
agent/skills.py            skill loader (progressive disclosure)
agent/triage.py            typed LLM ticket triage (priority and review rules in code)
skills/*/SKILL.md          incident-triage, root-cause-analysis, major-incident, outage-communication
mcp_server/server.py       MCP server: tickets, services, changes, metrics, logs, KB, SQL, fixes, pages
mcp_server/scenarios.py    demo worlds: cache-outage, major-incident
mcp_server/simulation.py   dynamics: clock, hidden faults, derived health (fixes vs. masks that relapse)
mcp_server/database.py     the simulated IT system (SQLite)
tests/                     3-level test pyramid
docs/presentation.md       thought process, flow, agentic AI concepts, testing strategy
```

## Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # add OPENAI_API_KEY (optionally OPENAI_MODEL)
streamlit run app.py
```

The terminal logs every step: the plan, each tool call and result (with timing, including subagent calls),
pauses for approval, human decisions, real internet calls, and MCP writes. Set `LOG_LEVEL=DEBUG` in `.env` to
see full tool results.

## Test

```bash
pytest            # 73 tests, scripted fake LLM, no API key, ~20 s
pytest -m live    # real internet checks + end-to-end scenarios with the real LLM
```

## Scenarios: a world with cause and effect

Pick one in the sidebar and click **Reset demo data** (or set `ITSM_SCENARIO`).

| Scenario | What is wrong | What a good agent does |
|---|---|---|
| `cache-outage` | web-shop fails because its cache is out of memory (one fault, one hop) | Finds the cache, flushes or restarts it, verifies, resolves T-101 |
| `major-incident` | A payment-api deploy (CHG-231) exhausts core-db connections: checkout, payments and logins fail (two hops). Red herrings: a web-shop deploy 15 min earlier, cache memory at 78%, an unrelated mailbox ticket | Groups T-201..T-204, runs diagnostics and change analysis in parallel, **rolls back CHG-231**. Restarting core-db looks like it works, then relapses after 3 min |

Health is derived from hidden faults ([simulation.py](mcp_server/simulation.py)). An action either fixes a fault
or only masks it for a while, and time only passes when the system observes. So "it looked fixed" and "it is fixed"
are different, and tests stay deterministic. The agent sees symptoms only: `run_sql` cannot read the `sim_*` tables.

## Planning, delegation and control

| Capability | How | Where |
|---|---|---|
| **Plan as state** | The planner's steps (3+) seed a todo list. The agent updates it with `write_todos`; the UI shows each version, so replanning is visible | [planning.py](agent/planning.py) |
| **Parallel specialists** | deepagents `SubAgentMiddleware` (`task` tool). Each subagent's steps are attributed exactly, even when they run at the same time (stream namespace → task id → subagent) | [subagents.py](agent/subagents.py) |
| **Working notes** | deepagents virtual filesystem: `/incident/notes.md` (hypotheses, evidence, timeline), readable by subagents | [agent.py](agent/agent.py) |
| **Gates before approval** | Code blocks a fix or a resolve while the last fix failed verification and the plan wasn't revised, and blocks resolving a ticket whose service is unhealthy. No human is asked for something that must not happen | [control.py](agent/control.py) |
| **Change critic** | A structured LLM review of each fix against the evidence: `supported` / `weak` go on the approval card; `contradicted` is sent back to the agent. If the critic fails, the human still decides | [critic.py](agent/critic.py) |
| **Verification** | After every fix, code watches every connected service for 5 simulated minutes. FAILED (e.g. a relapse) forces a revised plan before any other fix | [control.py](agent/control.py) |

The order of the `after_model` hooks is the design: tool-call cap → gates → critic → human approval.

## Long-term memory

Short-term memory (the checkpointer) keeps one conversation together. Long-term memory ([agent/memory.py](agent/memory.py))
carries what the desk learned into every later conversation. It is built on **deepagents** memory: a `StoreBackend`
over a SQLite LangGraph store (`data/memory.db`, kept by "Reset demo data").

| Memory | Written when | Used how |
|---|---|---|
| **Lessons** `/memories/AGENTS.md` | A human rejects an action with a reason, corrects the agent, or asks it to remember a rule (`save_lesson`) | Loaded into the planner's and main agent's prompt on **every turn** (`MemoryMiddleware`, reloaded so all open sessions see it) |
| **Past incidents** `/memories/incidents/*.md` | A fix is verified and the ticket resolved (`record_incident`, same approval as `update_ticket`) | Searched on demand by the agent and `it_diagnostics` (`search_past_incidents`) |

A memory outlives the conversation that wrote it, so a poisoned one would affect every future run. Guardrails, in
code: every memory write needs human approval (each pending action can be approved or rejected on its own);
writes are typed tools with fixed paths, fields and size caps, not free-form file edits; text that reads like
instructions to an AI is refused; lessons are capped at 30; memory is framed as reference data, never instructions.

Try it: ask to fix the web shop, reject the cache restart with *"Never restart cache in business hours, page
app-team"*, approve the lesson the agent proposes, then click **➕ New chat** and ask again: the plan now
pages app-team instead of restarting. Resolve T-101 in one chat, then ask about it in a new chat and watch
`it_diagnostics` find the past incident.
