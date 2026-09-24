# IT Service Desk Agent

An **agentic AI** system built with **LangChain**. An IT service-desk agent works tickets end to end:
- as soon as you ask, it replies with how it understood the request and its plan, then shows every step live;
- it answers simple questions with one direct lookup, and triages tickets with **Jev** (typed, calibrated decisions);
- it delegates multi-step investigation to **subagents**;
- it reads internal systems through an **MCP server**;
- it checks the **real internet** (websites, DNS, TLS, vendor status pages);
- it follows team procedures stored as **skills**;
- it fixes problems only **after a human approves**.

```
app.py                     Streamlit UI: chat, agent steps, approval card, live environment
agent/agent.py             main agent (LangChain create_agent + planning + memory + human-in-the-loop)
agent/subagents.py         it_diagnostics + internet_checker (agents wrapped as tools)
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
pytest            # 29 tests, scripted fake LLM + fake Jev, no API key, ~8 s
pytest -m live    # real internet checks + end-to-end scenarios with the real LLM
```
