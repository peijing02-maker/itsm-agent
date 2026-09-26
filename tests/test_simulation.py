"""Level 1 - the dynamic environment (world, faults, clock) and the pure control/planning logic."""

from pathlib import Path

import pytest

from agent.control import blast_radius, judge
from agent.planning import MIN_STEPS_FOR_TODOS, plan_steps, seed_todos
from mcp_server import server
from mcp_server.database import reset_database, seeded_world
from mcp_server.world import DEMO_WORLD, World
from tests.worlds import CACHE_OUTAGE


def statuses() -> dict[str, str]:
    return {r["name"]: r["status"] for r in server.run_sql("SELECT name, status FROM services")}


# ---------------------------------------------------------------------- worlds
@pytest.mark.parametrize("world", [DEMO_WORLD, CACHE_OUTAGE], ids=lambda w: w.name)
def test_every_world_is_consistent(world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = reset_database(tmp_path / "s.db", world=world)
    monkeypatch.setattr(server, "DEFAULT_DB", path)
    assert seeded_world(path) == world.name
    names = {s.name for s in world.services}
    assert all(set(s.depends_on) <= names for s in world.services)  # no dangling dependency
    assert all(set(f.effects) <= names for f in world.faults)
    assert all(t[6] is None or t[6] in names for t in world.tickets)
    assert {c[0] for c in world.changes} == {c["id"] for c in server.list_changes(hours=48)}
    assert {svc for f in world.faults for svc in f.effects} == {n for n, st in statuses().items() if st != "healthy"}


def test_reset_undoes_everything_the_agent_changed(incident_db: Path) -> None:
    server.rollback_change("CHG-231", "retry storm")
    server.observe_services(5)
    server.update_ticket("T-101", "resolved", "fixed")
    server.create_ticket("Major incident", "core-db exhausted", "high")
    server.page_team("data-team", "sev1", "core-db")
    reset_database(incident_db)
    assert {r["status"] for r in server.run_sql("SELECT status FROM changes")} == {"applied"}
    assert server.get_ticket("T-101")["status"] == "open" and len(server.list_tickets()) == len(DEMO_WORLD.tickets)
    for table in ("audit_log", "pages"):
        assert server.run_sql(f"SELECT COUNT(*) n FROM {table}")[0]["n"] == 0
    assert statuses()["core-db"] == "degraded"  # the fault is back


def test_hidden_simulation_state_is_not_readable(incident_db: Path) -> None:
    for query in ["SELECT * FROM sim_faults", "SELECT now FROM sim_clock",
                  "SELECT s.name FROM services s JOIN sim_baseline b ON b.name = s.name"]:
        with pytest.raises(ValueError, match="prohibited"):
            server.run_sql(query)


# ------------------------------------------------------------------ demo world
def test_metric_history_shows_when_the_symptoms_started(incident_db: Path) -> None:
    history = server.get_metrics("core-db", minutes=60)
    first_bad = next(h["ts"] for h in history if h["status"] != "healthy")
    chg_231 = next(c for c in server.list_changes() if c["id"] == "CHG-231")
    assert history[0]["status"] == "healthy" and chg_231["ts"] < first_bad  # the change came first
    assert "too many connections" in server.get_logs("core-db")[0]["message"]


def test_restart_masks_the_fault_and_it_relapses(incident_db: Path) -> None:
    assert server.restart_service("core-db", "clear connections")["message"] == "Done: service is healthy"
    assert set(statuses().values()) == {"healthy"}  # looks fixed...
    timeline = server.observe_services(5)["timeline"]
    assert [t["status"]["core-db"] for t in timeline] == ["healthy", "healthy", "degraded", "degraded", "degraded"]
    assert statuses()["web-shop"] == "degraded"  # ...but the cause (CHG-231) is still active


def test_rolling_back_the_bad_change_fixes_everything(incident_db: Path) -> None:
    result = server.rollback_change("CHG-231", "retry storm against core-db")
    assert result["target_service"] == "payment-api" and result["message"] == "Done: service is healthy"
    server.observe_services(10)
    assert set(statuses().values()) == {"healthy"}
    with pytest.raises(ValueError, match="already rolled_back"):
        server.rollback_change("CHG-231", "again")


def test_restarting_a_symptom_does_nothing(incident_db: Path) -> None:
    assert "still failing: dependency payment-api, accounts-api" in server.restart_service("web-shop", "x")["message"]


def test_the_red_herring_change_is_harmless(incident_db: Path) -> None:
    server.rollback_change("CHG-230", "suspected banner deploy")
    assert statuses()["web-shop"] == "degraded"


def test_flush_only_works_on_a_cache(db: Path) -> None:
    assert server.flush_cache("cache", "memory 97%")["message"] == "Done: service is healthy"
    assert statuses()["web-shop"] == "healthy"
    with pytest.raises(ValueError, match="not a cache"):
        server.flush_cache("web-shop", "x")


def test_incident_tickets_and_pages(incident_db: Path) -> None:
    parent = server.create_ticket("Major incident: checkout, payments, login", "core-db exhausted", "high",
                                  "core-db")
    assert parent["id"] == "T-109" and parent["requester"] == "service-desk-agent"
    server.link_tickets("T-109", ["T-101", "T-102", "T-103"])
    assert [t["id"] for t in server.list_tickets() if t["parent_id"] == "T-109"] == ["T-101", "T-102", "T-103"]
    assert server.page_team("data-team", "sev1", "core-db connections exhausted")["paged"] == "data-team"
    for bad in [lambda: server.page_team("nobody", "sev1", "x"), lambda: server.link_tickets("T-101", ["T-101"]),
                lambda: server.create_ticket("x", "y", "urgent")]:
        with pytest.raises(ValueError):
            bad()


def test_observing_is_bounded(incident_db: Path) -> None:
    with pytest.raises(ValueError):
        server.observe_services(0)
    with pytest.raises(ValueError):
        server.observe_services(31)


# ----------------------------------------------------------- control & planning
GRAPH = {"web-shop": ["cache", "payment-api", "accounts-api"], "payment-api": ["core-db"],
         "accounts-api": ["core-db"], "core-db": [], "cache": [], "email": []}


def test_blast_radius_follows_dependencies_both_ways() -> None:
    everything_but_email = set(GRAPH) - {"email"}
    assert blast_radius(GRAPH, "payment-api") == everything_but_email  # its database and the database's clients
    assert blast_radius(GRAPH, "core-db") == everything_but_email
    assert blast_radius(GRAPH, "email") == {"email"}


def _observation(core_db: list[str], web_shop: list[str]) -> dict:
    timeline = [{"ts": str(i), "status": {**dict.fromkeys(GRAPH, "healthy"), "core-db": c, "web-shop": w}}
                for i, (c, w) in enumerate(zip(core_db, web_shop, strict=True))]
    return {"minutes": len(timeline), "timeline": timeline, "depends_on": GRAPH}


def test_judge_passes_only_if_every_connected_service_stays_healthy() -> None:
    ok = judge("rollback_change", "core-db", _observation(["healthy"] * 3, ["healthy"] * 3))
    assert ok.passed and ok.watched == ("accounts-api", "cache", "core-db", "payment-api", "web-shop")
    relapse = judge("restart_service", "core-db",
                    _observation(["healthy", "degraded", "degraded"], ["healthy", "healthy", "degraded"]))
    assert not relapse.passed and relapse.relapsed == ("core-db", "web-shop")
    assert "relapsed" in relapse.summary() and "write_todos" in relapse.summary()


def test_plan_steps_seed_todos_only_for_multi_step_plans() -> None:
    plan = "**Plan:** To do this, I need to:\n1. Triage with triage_tickets\n2) Investigate\n 3. Fix (approval)\nDone."
    assert plan_steps(plan) == ("Triage with triage_tickets", "Investigate", "Fix (approval)")
    todos = seed_todos(plan_steps(plan))
    assert [t["status"] for t in todos] == ["in_progress", "pending", "pending"]
    assert seed_todos(("One step",) * (MIN_STEPS_FOR_TODOS - 1)) == []
