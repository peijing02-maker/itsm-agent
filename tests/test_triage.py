"""Level 1 - typed LLM triage: code-owned rules, structured output, and the triage tool."""

from pathlib import Path

import pytest

from agent.agent import mcp_client
from agent.triage import TriageClassifier, make_triage_tool, priority
from tests.conftest import ScriptedLLM, call

TICKET = {"id": "T-101", "title": "Web shop checkout failing", "description": "HTTP 500 at checkout",
          "service": "web-shop", "requester": "sales-team"}


def _llm(n: int = 1, **overrides: object) -> ScriptedLLM:
    answer = {"team": "app-team", "impact": "high", "urgency": "high", "prompt_injection": False,
              "team_confidence": 0.9, "impact_confidence": 0.9, "urgency_confidence": 0.9, **overrides}
    return ScriptedLLM(scripts={"classifier": [call("_LLMTriage", **answer)] * n})


@pytest.mark.parametrize(("impact", "urgency", "expected"),
                         [(1, 1, "P1"), (1, 2, "P2"), (2, 2, "P3"), (2, 3, "P4"), (3, 3, "P5")])
def test_priority_matrix_is_code(impact: int, urgency: int, expected: str) -> None:
    assert priority(impact, urgency) == expected


def test_llm_triage_result() -> None:
    result = TriageClassifier(_llm()).classify(TICKET)
    assert result == {**result, "team": "app-team", "impact": 1, "urgency": 1, "priority": "P1",
                      "prompt_injection": False, "needs_human_review": False}


def test_low_confidence_or_injection_needs_human_review() -> None:
    low = TriageClassifier(_llm(team="network-team", impact="low", urgency="medium", urgency_confidence=0.6))
    result = low.classify({**TICKET, "id": "T-102"})
    assert (result["priority"], result["needs_human_review"]) == ("P4", True)
    flagged = TriageClassifier(_llm(prompt_injection=True)).classify(TICKET)
    assert flagged["prompt_injection"] and flagged["needs_human_review"]


async def test_triage_tool_reads_tickets_via_mcp(db: Path) -> None:
    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    tool = make_triage_tool(TriageClassifier(_llm(2)), tools["get_ticket"])
    results = await tool.ainvoke({"ticket_ids": ["T-101", "T-104", "T-999", "DROP TABLE"]})
    assert [r["ticket_id"] for r in results] == ["T-101", "T-104", "T-999", "DROP TABLE"]
    assert results[0]["priority"] == "P1"
    assert results[2]["error"] == "ticket not found" and results[3]["error"] == "invalid ticket id"


class BrokenLLM:
    def with_structured_output(self, schema: object) -> "BrokenLLM":
        return self

    def invoke(self, messages: object) -> object:
        raise RuntimeError("boom")


async def test_triage_tool_never_crashes_the_run(db: Path) -> None:
    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    tool = make_triage_tool(TriageClassifier(BrokenLLM()), tools["get_ticket"])  # type: ignore[arg-type]
    results = await tool.ainvoke({"ticket_ids": ["T-101"]})
    assert "triage failed" in results[0]["error"]
