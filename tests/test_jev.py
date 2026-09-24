"""Level 1 - Jev typed triage: code-owned rules, the Jev path (fake client), and the LLM fallback."""

from pathlib import Path
from typing import Any

import pytest
import typesafe_sdk as ts

from agent.agent import mcp_client
from agent.jev import JevClassifier, make_jev_triage_tool, priority
from tests.conftest import ScriptedLLM, call

TICKET = {"id": "T-101", "title": "Web shop checkout failing", "description": "HTTP 500 at checkout",
          "service": "web-shop", "requester": "sales-team"}


class FakeJev:
    """Stands in for TypeSafeClient.system_one and records what was sent."""

    def __init__(self, team_conf: float = 0.93, injection: float = 0.02) -> None:
        self.requests: list[dict[str, Any]] = []
        self.team_conf, self.injection = team_conf, injection

    def system_one(self, state: Any, questions: Any) -> ts.SystemOneResponse:
        self.requests.append({"state": state, "questions": questions})
        legend = {0: "low", 1: "medium", 2: "high"}
        return ts.SystemOneResponse(model="jev-test", usage=ts.Usage(), answers={
            "team": ts.ChoiceAnswer(choice="app-team", confidence=self.team_conf, probabilities={"app-team": 0.93}),
            "impact": ts.ScoreAnswer(score=1.8, confidence=0.9, legend=legend, probabilities={0: 0.05, 1: 0.15, 2: 0.8}),
            "urgency": ts.ScoreAnswer(score=1.9, confidence=0.9, legend=legend, probabilities={0: 0.02, 1: 0.08, 2: 0.9}),
            "prompt_injection": ts.NoulAnswer(noul=self.injection),
        })


@pytest.mark.parametrize(("impact", "urgency", "expected"),
                         [(1, 1, "P1"), (1, 2, "P2"), (2, 2, "P3"), (2, 3, "P4"), (3, 3, "P5")])
def test_priority_matrix_is_code(impact: int, urgency: int, expected: str) -> None:
    assert priority(impact, urgency) == expected


def test_jev_one_request_four_typed_answers() -> None:
    fake = FakeJev()
    result = JevClassifier(client=fake).classify(TICKET)
    assert len(fake.requests) == 1 and set(fake.requests[0]["questions"]) == {"team", "impact", "urgency",
                                                                              "prompt_injection"}
    assert "requester" not in fake.requests[0]["state"]  # no personal data sent
    assert result == {**result, "team": "app-team", "impact": 1, "urgency": 1, "priority": "P1",
                      "prompt_injection": False, "needs_human_review": False, "decided_by": "jev"}


def test_low_confidence_or_injection_needs_human_review() -> None:
    assert JevClassifier(client=FakeJev(team_conf=0.55)).classify(TICKET)["needs_human_review"] is True
    flagged = JevClassifier(client=FakeJev(injection=0.97)).classify(TICKET)
    assert flagged["prompt_injection"] and flagged["needs_human_review"]


def test_llm_fallback_without_jev_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    llm = ScriptedLLM(scripts={"classifier": [call(
        "_LLMTriage", team="network-team", impact="low", urgency="medium", prompt_injection=False,
        team_confidence=0.9, impact_confidence=0.8, urgency_confidence=0.6)]})
    clf = JevClassifier(fallback_model=llm)
    assert clf.mode == "llm-fallback"
    result = clf.classify({**TICKET, "id": "T-102"})
    assert (result["priority"], result["decided_by"], result["needs_human_review"]) == ("P4", "llm-fallback", True)


async def test_jev_triage_tool_reads_tickets_via_mcp(db: Path) -> None:
    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    fake = FakeJev()
    tool = make_jev_triage_tool(JevClassifier(client=fake), tools["get_ticket"])
    results = await tool.ainvoke({"ticket_ids": ["T-101", "T-104", "T-999", "DROP TABLE"]})
    assert [r["ticket_id"] for r in results] == ["T-101", "T-104", "T-999", "DROP TABLE"]
    assert results[0]["priority"] == "P1"
    sent = [r["state"]["description"] for r in fake.requests]  # tickets are classified concurrently: any order
    assert len(sent) == 2 and any("IGNORE ALL PREVIOUS" in d for d in sent)
    assert results[2]["error"] == "ticket not found" and results[3]["error"] == "invalid ticket id"


class FailingJev:
    def __init__(self, error: Exception) -> None:
        self.error, self.calls = error, 0

    def system_one(self, state: Any, questions: Any) -> Any:
        self.calls += 1
        raise self.error


def _fallback_llm(n: int) -> ScriptedLLM:
    answer = call("_LLMTriage", team="app-team", impact="high", urgency="high", prompt_injection=False,
                  team_confidence=0.9, impact_confidence=0.9, urgency_confidence=0.9)
    return ScriptedLLM(scripts={"classifier": [answer] * n})


def test_bad_key_disables_jev_and_falls_back_to_llm() -> None:
    import httpx2

    failing = FailingJev(ts.TypeSafeAuthenticationError(401, None, httpx2.Headers()))
    clf = JevClassifier(client=failing, fallback_model=_fallback_llm(2))
    assert clf.classify(TICKET)["decided_by"] == "llm-fallback"
    assert clf.mode == "llm-fallback" and "TYPESAFE_API_KEY" in (clf.disabled_reason or "")
    clf.classify(TICKET)
    assert failing.calls == 1  # no repeated failing calls after an auth error


def test_transient_jev_error_falls_back_for_that_ticket_only() -> None:
    import httpx2

    failing = FailingJev(ts.TypeSafeRateLimitError(429, None, httpx2.Headers()))
    clf = JevClassifier(client=failing, fallback_model=_fallback_llm(1))
    assert clf.classify(TICKET)["decided_by"] == "llm-fallback"
    assert clf.mode == "jev"  # still enabled: rate limits are temporary


async def test_triage_tool_never_crashes_the_run(db: Path) -> None:
    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    broken = JevClassifier(client=FailingJev(RuntimeError("boom")))  # unexpected error, no fallback
    results = await make_jev_triage_tool(broken, tools["get_ticket"]).ainvoke({"ticket_ids": ["T-101"]})
    assert "triage failed" in results[0]["error"]


async def test_bad_key_is_detected_once_not_per_ticket(db: Path) -> None:
    import httpx2

    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    failing = FailingJev(ts.TypeSafeAuthenticationError(401, None, httpx2.Headers()))
    clf = JevClassifier(client=failing, fallback_model=_fallback_llm(4))
    results = await make_jev_triage_tool(clf, tools["get_ticket"]).ainvoke({"ticket_ids": ["T-101", "T-102", "T-103", "T-104"]})
    assert failing.calls == 1 and all(r["decided_by"] == "llm-fallback" for r in results)
