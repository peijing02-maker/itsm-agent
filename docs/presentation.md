# IT Service Desk Agent: presentation notes

## 1. What I built

An **IT Service Desk agent** built with LangChain. You give it a goal ("Investigate T-101, fix the root cause
and resolve it") and it works autonomously:

- it delegates investigation to specialist **subagents**;
- it reads internal IT systems through an **MCP server**;
- it checks **real internet systems**;
- it follows team procedures stored as **skills**;
- it changes things only after a **human approves**.

## 2. Architecture

```
                      ┌──────────────────────── Streamlit UI (app.py) ───────────────────────┐
                      │ chat · agent steps · approval card · live services/tickets/audit log │
                      └───────────────────────────────┬──────────────────────────────────────┘
                                                      │
                     ┌──── 1. PLAN: planner LLM call, streamed at once ("You're asking about... I need to...") ─┐
                     └───────────────────────────────────────────┬───────────────────────────────────────────┘
                     ┌──── 2. EXECUTE: service_desk_lead (LangChain create_agent) ──────────────────────────────┐
                     │ LLM · memory (checkpointer) · HITL middleware · every step streamed live to the UI       │
                     └──┬──────────────────┬──────────────────┬───────────────┬──────────────────────────────────┘
                        │ subagent         │ subagent         │ skills        │ write tools (need approval)
              ┌─────────▼─────────┐ ┌──────▼───────────┐ ┌────▼──────┐  restart_service, update_ticket
              │ it_diagnostics    │ │ internet_checker │ │ load_skill│             │
              │ (read-only)       │ │                  │ │ SKILL.md  │             │
              └─────────┬─────────┘ └──────┬───────────┘ └───────────┘             │
                        │ MCP (stdio)      │ real HTTP / DNS / TLS                  │ MCP (stdio)
              ┌─────────▼──────────────────┼────────────────────────────────────────▼──┐
              │ MCP server "itsm": tickets · services · knowledge base · SQL · restart │
              └─────────────────────────────────┬──────────────────────────────────────┘
                                                ▼
                                   SQLite IT system (simulated)
```

## 3. Showcase features

| Feature | What it shows | Where |
|---|---|---|
| **LangChain agent** | `create_agent` with a model, tools and middleware: the standard ReAct loop (reason, act, observe), preceded by a planner step | `agent/agent.py` |
| **Subagents** | Specialists with their own prompt, tools and clean context. The main agent only sees their short report. | `agent/subagents.py` |
| **MCP** | Tools served over the Model Context Protocol. The server is framework-independent and could also be used by Claude Desktop, Cursor or other clients. | `mcp_server/server.py` |
| **Skills** | Procedures stored as markdown. Only the name and description go in the prompt; the full text is loaded on demand. New skills need no code changes. | `skills/*/SKILL.md`, `agent/skills.py` |
| **Real tools** | Live website check, DNS lookup, TLS certificate expiry, and SaaS vendor status (GitHub, OpenAI, Cloudflare...) | `agent/real_tools.py` |
| **Jev (System One)** | Fast typed decisions (choice, score, yes/no) with calibrated confidence: ticket team, impact, urgency, prompt-injection flag. Priority is computed in code; low confidence goes to a human. Falls back to the LLM without a key. | `agent/jev.py` |
| **Human-in-the-loop** | Write tools pause the graph; the human approves or rejects | `HumanInTheLoopMiddleware` |
| **Transparency (plan-and-execute)** | The user sees the understanding and plan within ~1.5 s, then every step live | `astream_chat` in `agent/agent.py`, `stream_turn` in `app.py` |

## 4. Demo script (5–7 minutes)

In every demo, the agent first streams **how it understood the request and its plan**. Then each tool call
appears live, with subagent calls indented under the subagent that made them.

1. **"Triage all open tickets."** The agent loads the `incident-triage` skill, delegates to `it_diagnostics`, and
   returns priorities and owners. It flags T-104 as a prompt injection.
2. **"Investigate T-101, fix the root cause and resolve it."** This is the main demo:
   - The plan appears first. Then `it_diagnostics` finds that web-shop depends on cache, and cache memory is at 97%.
   - It loads the `root-cause-analysis` skill, which says to restart the root cause and not the symptom.
   - It asks to run `restart_service(cache)`. **An approval card appears**, and I approve.
   - It verifies the fix, then asks to run `update_ticket(T-101, resolved)`, and I approve.
   - The sidebar shows services, tickets and the audit log changing live.
3. **"Users say GitHub is down. Is it us or them?"** `internet_checker` checks the real githubstatus.com,
   website reachability and DNS. These are live results.
4. **"Check the TLS certificate and DNS of openai.com."** Real certificate expiry and real IP addresses.
5. **Reject an action.** The agent does not retry and explains the alternatives.

## 5. Thought process and implementation flow

1. **Pick a domain where agents add value.** Service-desk work is multi-step (read, investigate, act, verify) and
   mixes internal systems with the outside world, and mistakes are costly. That makes it a good fit for
   autonomy plus guardrails.
2. **Environment first.** A small simulated IT system with *cause and effect*: web-shop fails because the cache is
   full. Restarting web-shop doesn't help; restarting the cache fixes both. This tests reasoning, not only tool
   calling.
3. **Expose the environment through MCP.** Tools become a reusable, standard interface instead of being tied to
   one framework.
4. **Split responsibilities.**
   - The main agent plans, decides and acts.
   - The subagents investigate: one is internal and read-only, one covers the internet.
   - Skills hold procedures.
   - This keeps each prompt and context small and focused.
5. **Put safety in code, not prompts.** Write tools always need approval (middleware), and SQL is read-only in
   the server. Prompts can be ignored or injected; code can't.
6. **Test at three levels** (section 8).

Flow of one request (plan-and-execute; everything streams to the UI as it happens):
```
user goal → PLANNER streams: "You're asking about T-101... To do this, I need to: 1. ... 2. ..."  (~1.5 s)
          → the plan is added to the conversation, and the EXECUTOR agent follows it:
          → load_skill(root-cause-analysis)
          → it_diagnostics subagent ──► MCP: get_ticket, check_service ×2, search_knowledge_base → report
          → restart_service(cache)  ──► PAUSE ──► human approves ──► MCP executes
          → it_diagnostics: verify web-shop healthy
          → update_ticket(resolved) ──► PAUSE ──► approve
          → final answer: findings, actions, evidence
```

## 5b. Efficiency: using only the tools a request needs

Measured with the real model (tool calls, LLM calls, time). Before is the first version; after adds the fixes below:

| Request | Before | After |
|---|---|---|
| "How many open tickets?" | 2 / 4, 9 s | **1 / 2, 4 s** |
| "Status of the cache?" | 14 / 9, 55 s, and it *tried to restart the cache* | **1 / 2, 4 s**, no action |
| "Triage all open tickets" | 15 / 6, 41 s | **3 / 3** (Jev triage) |
| "Fix T-101" end to end | 24 / ~21, ~80 s | **12 / 9, ~27 s** |

What changed:
1. **Do only what was asked.** Questions get answers, not actions, and the planner scales the number of steps
   to the request.
2. **Right tool for the job.** Simple facts use one direct MCP lookup. Triage uses one Jev call. Only multi-step
   root-cause work goes to a subagent. Delegating has a cost (extra LLM calls), so it should earn its place.
3. **Efficiency rules for subagents:** stop when the evidence is enough, don't repeat searches, don't query
   history unless asked.
4. **Hard caps in code:** `ToolCallLimitMiddleware` (10 per request for the main agent, 6 per subagent task).
5. **Measure it:** every request logs `DONE in Xs: N tool calls (main / subagents), M LLM calls`.

## 6. Agentic AI: core components

| Component | Role | In this project |
|---|---|---|
| **LLM (brain)** | Reasons, plans, chooses the next action | OpenAI model via `init_chat_model` |
| **Tools (actions)** | Act on the environment | MCP tools + real internet tools |
| **Environment** | The world the agent perceives and changes | SQLite IT system + the internet |
| **Planning** | Break goals into steps | Explicit planner step (plan-and-execute), shown to the user first, then adapted during the ReAct loop |
| **Memory** | Keep context, learn | Short-term: checkpointer per thread. Long-term (deepagents `StoreBackend` + `MemoryMiddleware`): human-approved lessons in every prompt, past incidents searched on demand. Plus skills and the knowledge base |
| **Orchestration** | Run the loop; coordinate agents | LangChain / LangGraph agent loop; subagents as tools |
| **Guardrails** | Keep autonomy safe | Human-in-the-loop, read-only SQL, tool-call caps, Jev injection flag, injection-aware prompts |
| **Fast decisions (System One)** | Typed, calibrated classification | Jev: team, impact, urgency, injection, with a confidence each |

## 7. Agentic AI: key characteristics

- **Autonomy:** the user states a goal, and the agent decides on 8–12 steps itself.
- **Goal-directed:** it continues until the goal is verified, not after a single reply.
- **Tool use:** it reads *and changes* real state (internal systems and live internet checks).
- **Reasoning and adaptivity:** it follows dependencies to the root cause, and adapts to errors and rejections.
- **Reflection / verification:** it re-checks after acting.
- **Collaboration:** a main agent plus specialist subagents.
- **Human oversight:** bounded autonomy.

Risks and how they are handled: hallucinated actions (tool results are evidence, and a human approves), prompt
injection (ticket text is treated as data, and approvals are enforced in code), runaway loops (recursion
limit), cost and latency (small subagent contexts, skills loaded only on demand).

## 8. Test cases to assure quality (test pyramid)

| Level | File | LLM | What it proves |
|---|---|---|---|
| 1. Components | `tests/test_mcp_server.py` | none | Tool logic is correct. SQL is read-only. The environment has realistic cause and effect. Tools work over the **real MCP protocol**. |
| | `tests/test_real_tools.py` | none | Internet tools handle up, down and unknown cases (network mocked). A live check hits real GitHub, DNS and TLS. |
| | `tests/test_jev.py` | **fake Jev client** | Priority matrix is code. One Jev request returns four typed answers. No personal data is sent. Low confidence or injection goes to human review. LLM fallback works. Tool reads tickets via MCP and rejects bad ids. |
| | `tests/test_skills.py` | none | Skills are discoverable; only descriptions go in the prompt, and the full text loads on demand. |
| 2. Agent behaviour | `tests/test_agent.py`, `tests/test_app.py` | **scripted fake** | The real LangChain graph with real MCP: the plan is streamed before any tool runs, subagent steps are streamed, write actions **pause until approved**, tool-call caps stop runaway subagents, simple questions use one direct tool, rejected actions never run, skills load on demand, memory works within a thread, injection cannot bypass approval, and the UI flow works (plan, steps, approval card, answer). |
| | `tests/test_memory.py` | **scripted fake** | Memory survives a restart; poisoned, oversized and duplicate writes are refused; lessons are capped; parallel writes keep every lesson; incidents are found by relevance. In the agent: nothing is remembered before approval; an approved lesson reaches the planner and executor of *other open sessions*; `it_diagnostics` reuses a past incident; the fix and the memory write are approved separately. |
| 3. End to end | `tests/test_scenarios.py` | **real** (`-m live`) | Real tasks: the root cause is fixed and the ticket resolved, it investigated *before* acting, real internet checks were used, the injection was ignored, and rejections were respected. |

Testing a non-deterministic system:
- **Assert outcomes and behaviour, not wording:** the final environment state and the tools used, in order.
- **Script the LLM** to test control flow deterministically, quickly and for free. Levels 1–2 take about 3 s.
- **Keep guardrails in code** so they are unit-testable regardless of model behaviour.
- **Next steps:** run live scenarios N times and report the pass rate, add an LLM-as-judge for answer quality,
  and track tokens, cost and latency.

## 9. Limitations and next steps

- The IT system is simulated. A ServiceNow or Jira MCP server would plug in with no agent changes.
- Long-term memory: move incident search to a store with an embedding index (semantic search) as history grows;
  per-operator memory once the UI has sign-in; feed triage corrections back into Jev's LLM fallback.
- Tracing with LangSmith, and CI running levels 1–2 on every commit.
