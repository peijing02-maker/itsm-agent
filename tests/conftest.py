"""Shared fixtures: fresh world databases and a scripted fake LLM (no network, no API key)."""

import itertools
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

import mcp_server.server as server
from mcp_server.database import reset_database
from tests.worlds import CACHE_OUTAGE

_ids = itertools.count(1)

# Which agent is calling? Identified by a phrase from its system prompt.
AGENT_MARKERS = {
    "planning step": "planner",
    "ticket triage classifier": "classifier",
    "lead IT Service Desk agent": "main",
    "IT diagnostics specialist": "it_diagnostics",
    "internet checker": "internet_checker",
    "change analyst": "change_analyst",
    "change critic": "critic",
}


def call(name: str, /, **args: Any) -> AIMessage:
    """Scripted LLM turn that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}", "type": "tool_call"}])


def delegate(subagent: str, task: str) -> AIMessage:
    """Scripted LLM turn that delegates to a subagent through the `task` tool."""
    return call("task", subagent_type=subagent, description=task)


def together(*turns: AIMessage) -> AIMessage:
    """Scripted LLM turn that makes several tool calls in one step (they run in parallel)."""
    return AIMessage(content="", tool_calls=[tc for t in turns for tc in t.tool_calls])


def verdict(value: str, reason: str) -> AIMessage:
    """Scripted change-critic answer (structured output arrives as a tool call)."""
    return call("CriticVerdict", verdict=value, reason=reason)


def say(text: str) -> AIMessage:
    """Scripted LLM turn with a final answer."""
    return AIMessage(content=text)


class ScriptedLLM(BaseChatModel):
    """Replays a script per agent, so main-agent and subagent turns can interleave."""

    scripts: dict[str, list[AIMessage]]
    _pos: dict[str, int] = PrivateAttr(default_factory=dict)
    _prompts: list[tuple[str, str]] = PrivateAttr(default_factory=list)  # (agent, system prompt) per call

    def prompts(self, agent: str) -> list[str]:
        """System prompts this agent was called with, in order."""
        return [text for who, text in self._prompts if who == agent]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedLLM":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        agent = next(a for marker, a in AGENT_MARKERS.items() if marker in system)
        self._prompts.append((agent, system))
        i = self._pos.get(agent, 0)
        self._pos[agent] = i + 1
        script = self.scripts.get(agent, [])
        msg = script[i] if i < len(script) else say("(script exhausted)")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture(autouse=True)
def memory_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Every test gets its own long-term memory; tests never touch data/memory.db."""
    path = tmp_path / "memory.db"
    monkeypatch.setenv("ITSM_MEMORY_DB_PATH", str(path))
    return path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The minimal test world (tests/worlds.py): web-shop fails because the cache is out of memory."""
    path = reset_database(tmp_path / "itsm.db", world=CACHE_OUTAGE)
    monkeypatch.setattr(server, "DEFAULT_DB", path)  # in-process calls
    monkeypatch.setenv("ITSM_DB_PATH", str(path))  # MCP subprocess
    return path


@pytest.fixture
def incident_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The demo world the app runs: payment-api deploy CHG-231 exhausts core-db connections."""
    path = reset_database(tmp_path / "itsm.db")
    monkeypatch.setattr(server, "DEFAULT_DB", path)
    monkeypatch.setenv("ITSM_DB_PATH", str(path))
    return path
