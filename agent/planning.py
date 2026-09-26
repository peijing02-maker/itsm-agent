"""The plan as state: the planner's numbered steps seed a todo list that the agent keeps up to date.

The planner streams a readable plan first (fast feedback for the user). For multi-step requests, its steps
become the todo list of LangChain's TodoListMiddleware (`write_todos`), so progress and replanning are state the
UI can show, not prose the model may forget. Simple requests get no todo list and cost no extra calls.
"""

import re
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, TodoListMiddleware
from langchain.agents.middleware.todo import PlanningState, Todo
from langgraph.runtime import Runtime

MIN_STEPS_FOR_TODOS = 3  # below this, a todo list only costs tool calls
_STEP = re.compile(r"^\s*\d+[.)]\s+(.+?)\s*$", re.MULTILINE)

TODO_PROMPT = """## Plan tracking (`write_todos`)
For multi-step requests, your todo list starts as the plan you wrote. Keep it true:
- When evidence changes the plan (a hypothesis is refuted, a fix fails verification), rewrite the list with
  write_todos BEFORE acting on the new plan: add, drop or reword steps.
- Mark steps completed as you go, but batch updates with other tool calls; never call write_todos twice in
  a row. Requests with 1-2 steps need no todo list."""


@dataclass(frozen=True)
class TurnContext:
    """Run-scoped input for one user request (LangGraph runtime context)."""

    plan_steps: tuple[str, ...] = ()


def plan_steps(plan: str) -> tuple[str, ...]:
    """The numbered steps of a plan written in the planner's format ('1. ...')."""
    return tuple(m.group(1) for m in _STEP.finditer(plan))


def seed_todos(steps: tuple[str, ...]) -> list[Todo]:
    """Todos for a new request: the first step in progress, the rest pending (none for short plans)."""
    if len(steps) < MIN_STEPS_FOR_TODOS:
        return []
    return [Todo(content=s, status="in_progress" if i == 0 else "pending") for i, s in enumerate(steps)]


class PlanSeedMiddleware(AgentMiddleware[PlanningState]):
    """At the start of each request, replace the previous request's todos with the new plan's steps."""

    state_schema = PlanningState

    def before_agent(self, state: PlanningState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        if not isinstance(runtime.context, TurnContext):  # e.g. resuming after an approval: keep the todos
            return None
        return {"todos": seed_todos(runtime.context.plan_steps)}

    async def abefore_agent(self, state: PlanningState, runtime: Runtime[Any]) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)


def planning_middleware() -> list[AgentMiddleware]:
    return [PlanSeedMiddleware(), TodoListMiddleware(system_prompt=TODO_PROMPT)]
