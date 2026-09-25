# IT Service Desk Agent

An **agentic AI** system built with **LangChain**. An IT service-desk agent works tickets end to end:
- as soon as you ask, it replies with how it understood the request and its plan, then shows every step live;
- it answers simple questions with one direct lookup, and triages tickets with **Jev** (typed, calibrated decisions);
- it delegates multi-step investigation to **subagents**;
- it reads internal systems through an **MCP server**;
- it checks the **real internet** (websites, DNS, TLS, vendor status pages);
- it follows team procedures stored as **skills**;
- it fixes problems only **after a human approves**;
- it **remembers** across conversations: lessons from human feedback and resolved incidents (a human approves
  every memory write).

```
app.py                     Streamlit UI: chat, agent steps, approval card, live environment
agent/agent.py             main agent (LangChain create_agent + planning + memory + human-in-the-loop)
agent/subagents.py         it_diagnostics + internet_checker (agents wrapped as tools)
agent/memory.py            long-term memory on deepagents: lessons in every prompt, past incidents on demand
agent/real_tools.py        real internet tools: website check, DNS, TLS expiry, vendor status
agent/skills.py            skill loader (progressive disclosure)
agent/jev.py               Jev (TypeSafe System One) typed ticket triage + LLM fallback
skills/*/SKILL.md          incident-triage, root-cause-analysis, outage-communication
mcp_server/server.py       MCP server: tickets, services, knowledge base, SQL, restart, update
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
pytest            # 45 tests, scripted fake LLM + fake Jev, no API key, ~8 s
pytest -m live    # real internet checks + end-to-end scenarios with the real LLM
```

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
