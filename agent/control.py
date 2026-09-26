"""Closed-loop control around production changes, enforced in code rather than in the prompt.

    ActionGateMiddleware    Runs after the model proposes actions and BEFORE a human is asked. It blocks
                            (1) fixes and ticket resolution while the last fix failed verification and the plan
                            has not been revised with write_todos, and (2) resolving a ticket whose service is
                            still unhealthy. Blocked calls return to the model as errors; no human is bothered.
    VerificationMiddleware  Runs after a fix (restart, flush, rollback). It lets simulated time pass
                            (VERIFY_MINUTES), watches every service connected to the fixed one, and
                            appends a PASSED/FAILED verdict to the tool result. FAILED sets `needs_replan`, which
                            only write_todos clears: the agent must revise its plan before trying anything else.

Why in code: a model told to "verify, and rethink if it fails" usually does. A system that cannot resolve an
unverified fix always does.
"""

import logging
import operator
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Annotated, Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent.mcp_utils import mcp_json
from mcp_server.server import REMEDIATION_TOOLS

log = logging.getLogger("itsm.control")

VERIFY_MINUTES = 5  # simulated minutes a fix must hold (a masked fault relapses within this window)
REPLAN_TOOL = "write_todos"

REPLAN_BLOCK = ("Blocked before approval: your last fix FAILED verification. Revise the plan first: call "
                "write_todos with the updated steps (what the failure tells you, the new hypothesis, the next "
                "action). Then propose the action again.")


class ControlState(AgentState):
    needs_replan: NotRequired[bool]
    verifications: Annotated[NotRequired[list[dict[str, Any]]], operator.add]


# ----------------------------------------------------------------- helpers
def pending_calls(messages: Sequence[AnyMessage]) -> tuple[AIMessage | None, list[ToolCall]]:
    """The last AI message and its tool calls that have no result yet (e.g. not blocked by a gate)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            ai = messages[i]
            answered = {m.tool_call_id for m in messages[i + 1:] if isinstance(m, ToolMessage)}
            return ai, [c for c in ai.tool_calls if c["id"] not in answered]
    return None, []


def is_answered(request: ToolCallRequest) -> bool:
    """HITL `when` predicate helper: True if a gate already answered this call (no human needed)."""
    _, calls = pending_calls(request.state["messages"])
    return all(c["id"] != request.tool_call["id"] for c in calls)


def blocked(call: ToolCall, reason: str) -> ToolMessage:
    return ToolMessage(reason, tool_call_id=call["id"], name=call["name"], status="error")


def blast_radius(depends_on: Mapping[str, Iterable[str]], service: str) -> set[str]:
    """Every service connected to `service` through dependencies, in either direction.

    A fix can matter upstream and downstream: rolling back a client's deploy heals the database it overloaded
    (a dependency) and every other client of that database (siblings), not only the client's dependents.
    """
    edges: dict[str, set[str]] = {name: set() for name in depends_on}
    for name, deps in depends_on.items():
        for dep in deps:
            edges[name].add(dep)
            edges.setdefault(dep, set()).add(name)
    found, frontier = {service}, {service}
    while frontier:
        frontier = {n for s in frontier for n in edges.get(s, ())} - found
        found |= frontier
    return found


@dataclass(frozen=True)
class Verification:
    action: str
    target_service: str
    passed: bool
    minutes: int
    watched: tuple[str, ...]
    unhealthy: dict[str, str]  # watched services not healthy at the end of the window
    relapsed: tuple[str, ...]  # healthy during the window, unhealthy again at the end

    def summary(self) -> str:
        head = (f"VERIFICATION ({self.minutes} min observed, in code): "
                f"{'PASSED' if self.passed else 'FAILED'}. Watched: {', '.join(self.watched)}.")
        if self.passed:
            return f"{head} All healthy for the whole window. Do not re-check them."
        detail = ", ".join(f"{s} is {st}" for s, st in self.unhealthy.items())
        relapse = (f" {', '.join(self.relapsed)} recovered at first, then relapsed: the action only relieved a "
                   "symptom; the cause is still active." if self.relapsed else "")
        return f"{head} {detail}.{relapse} The fix did not hold. Revise the plan with write_todos before acting."


def judge(action: str, target_service: str, observation: Mapping[str, Any]) -> Verification:
    """Verdict on a fix, from an observation window (pure: easy to test)."""
    timeline = observation["timeline"]
    watched = tuple(sorted(blast_radius(observation["depends_on"], target_service)))
    end = timeline[-1]["status"]
    unhealthy = {s: end[s] for s in watched if end[s] != "healthy"}
    relapsed = tuple(s for s in unhealthy if any(t["status"][s] == "healthy" for t in timeline))
    return Verification(action, target_service, not unhealthy, observation["minutes"], watched, unhealthy, relapsed)


# -------------------------------------------------------------- middleware
class ActionGateMiddleware(AgentMiddleware[ControlState]):
    """Deterministic gates between the model's proposal and the human approval card."""

    state_schema = ControlState

    def __init__(self, get_ticket: BaseTool, check_service: BaseTool) -> None:
        super().__init__()
        self.get_ticket, self.check_service = get_ticket, check_service

    async def aafter_model(self, state: ControlState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        _, calls = pending_calls(state["messages"])
        replanning = any(c["name"] == REPLAN_TOOL for c in calls)
        refusals = []
        for call in calls:
            resolves = call["name"] == "update_ticket" and call["args"].get("status") == "resolved"
            if state.get("needs_replan") and not replanning and (call["name"] in REMEDIATION_TOOLS or resolves):
                refusals.append(blocked(call, REPLAN_BLOCK))
            elif resolves and (problem := await self._unhealthy(call["args"].get("ticket_id", ""))):
                refusals.append(blocked(call, f"Blocked before approval: cannot resolve while {problem}. Fix and "
                                              "verify first, or set the ticket to in_progress with a note."))
        for msg in refusals:
            log.warning("GATE blocked %s: %s", msg.name, msg.text[:120])
        return {"messages": refusals} if refusals else None

    async def _unhealthy(self, ticket_id: str) -> str | None:
        """Why the ticket's service is not fit to resolve, or None (also None if the lookup fails)."""
        try:
            service = mcp_json(await self.get_ticket.ainvoke({"ticket_id": ticket_id})).get("service")
            if not service:
                return None
            status = mcp_json(await self.check_service.ainvoke({"service": service}))["status"]
        except (ValueError, TypeError, KeyError):
            return None  # unknown ticket/service: the tool itself reports the error
        return None if status == "healthy" else f"its service {service} is {status}"


class VerificationMiddleware(AgentMiddleware[ControlState]):
    """After every fix, observe the environment and put a verdict in front of the model."""

    state_schema = ControlState

    def __init__(self, observe: BaseTool, minutes: int = VERIFY_MINUTES) -> None:
        super().__init__()
        self.observe, self.minutes = observe, minutes

    async def awrap_tool_call(self, request: ToolCallRequest,
                              handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]
                              ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        result = await handler(request)
        if name == REPLAN_TOOL and request.state.get("needs_replan") and isinstance(result, Command):
            log.info("Plan revised after a failed verification: fixes are allowed again")
            return Command(update={**(result.update or {}), "needs_replan": False})
        if name not in REMEDIATION_TOOLS or not isinstance(result, ToolMessage) or result.status == "error":
            return result
        try:
            target = mcp_json(result.content)["target_service"]
            observation = mcp_json(await self.observe.ainvoke({"minutes": self.minutes}))
        except (ValueError, TypeError, KeyError) as exc:
            log.warning("Verification of %s skipped: %s", name, exc)
            return result
        verdict = judge(name, target, observation)
        (log.info if verdict.passed else log.warning)("%s", verdict.summary())
        update: dict[str, Any] = {
            "messages": [result.model_copy(update={"content": f"{result.text}\n\n{verdict.summary()}"})],
            "verifications": [asdict(verdict)],
        }
        if not verdict.passed:  # only a revised plan (write_todos) clears it
            update["needs_replan"] = True
        return Command(update=update)
