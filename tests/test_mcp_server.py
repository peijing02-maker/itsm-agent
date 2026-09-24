"""Level 1 - the MCP server (the agent's environment): tool logic + the real MCP protocol."""

from pathlib import Path

import pytest

from agent.agent import mcp_client
from mcp_server import server


def test_read_tools(db: Path) -> None:
    assert [t["id"] for t in server.list_tickets("open")] == ["T-101", "T-102", "T-103", "T-104"]
    assert server.get_ticket("T-101")["service"] == "web-shop"
    assert server.search_knowledge_base("web shop 500 errors")[0]["id"] == "KB-1"
    assert "cache" in server.check_service("web-shop")["depends_on"]
    with pytest.raises(ValueError):
        server.get_ticket("T-999")


def test_sql_answers_questions_but_is_read_only(db: Path) -> None:
    rows = server.run_sql("SELECT priority, COUNT(*) n FROM tickets WHERE status='open' GROUP BY priority")
    assert {r["priority"]: r["n"] for r in rows} == {"high": 1, "medium": 2, "low": 1}
    for bad in ["DELETE FROM tickets", "UPDATE tickets SET status='x'", "SELECT 1; DROP TABLE tickets"]:
        with pytest.raises(ValueError):
            server.run_sql(bad)
    assert len(server.list_tickets()) == 5


def test_restart_symptom_vs_root_cause(db: Path) -> None:
    assert "still failing" in server.restart_service("web-shop", "try")["message"]
    assert server.restart_service("cache", "memory 97%")["service_after"]["status"] == "healthy"
    assert server.check_service("web-shop")["status"] == "healthy"  # dependent recovered
    assert [r["target"] for r in server.run_sql("SELECT target FROM audit_log")] == ["web-shop", "cache"]


def test_update_ticket(db: Path) -> None:
    assert server.update_ticket("T-101", "resolved", "fixed")["status"] == "resolved"
    with pytest.raises(ValueError):
        server.update_ticket("T-101", "deleted", "x")


async def test_tools_over_real_mcp_protocol(db: Path) -> None:
    tools = {t.name: t for t in await mcp_client(db).get_tools()}
    assert set(tools) == set(server.READ_ONLY_TOOLS) | set(server.WRITE_TOOLS)
    assert all(t.description for t in tools.values())  # docstrings become tool descriptions
    result = await tools["check_service"].ainvoke({"service": "cache"})
    assert '"memory_pct": 97.0' in str(result)
    error = await tools["run_sql"].ainvoke({"query": "DELETE FROM tickets"})
    assert "only a single SELECT" in str(error)  # errors reach the LLM as text
