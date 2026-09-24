"""Shared fixtures: a fresh demo database and a scripted fake LLM (no network, no API key)."""

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

_ids = itertools.count(1)

# Which agent is calling? Identified by a phrase from its system prompt.
AGENT_MARKERS = {
    "planning step": "planner",
    "ticket triage classifier": "classifier",
    "lead IT Service Desk agent": "main",
    "IT diagnostics specialist": "it_diagnostics",
    "internet checker": "internet_checker",
}


def call(name: str, /, **args: Any) -> AIMessage:
    """Scripted LLM turn that calls one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}", "type": "tool_call"}])


def say(text: str) -> AIMessage:
    """Scripted LLM turn with a final answer."""
    return AIMessage(content=text)


class ScriptedLLM(BaseChatModel):
    """Replays a script per agent, so main-agent and subagent turns can interleave."""

    scripts: dict[str, list[AIMessage]]
    _pos: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedLLM":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any) -> ChatResult:
        system = next((m.text for m in messages if isinstance(m, SystemMessage)), "")
        agent = next(a for marker, a in AGENT_MARKERS.items() if marker in system)
        i = self._pos.get(agent, 0)
        self._pos[agent] = i + 1
        script = self.scripts.get(agent, [])
        msg = script[i] if i < len(script) else say("(script exhausted)")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = reset_database(tmp_path / "itsm.db")
    monkeypatch.setattr(server, "DEFAULT_DB", path)  # in-process calls
    monkeypatch.setenv("ITSM_DB_PATH", str(path))  # MCP subprocess
    return path
