"""The simulated IT environment's dynamics: a clock, hidden faults, and service health derived from them.

Service health is never edited directly. It is *derived*: every service starts from its healthy baseline, and
each active fault overlays its effects (on the faulty service and on the services that depend on it).
Actions change faults, not metrics:
    fixed_by   the fault is gone for good (e.g. roll back the change that caused it)
    masked_by  the symptoms disappear, and come back after `relapse_minutes` (e.g. restart a database whose
               connections a bad client keeps exhausting)
Time only moves when someone observes (`advance`), so runs are deterministic and tests need no sleeping.

The clock and faults live in `sim_*` tables, which the agent's SQL tool cannot read.
"""

import json
import math
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

TS_FORMAT = "%Y-%m-%d %H:%M:%S"  # UTC, same as SQLite CURRENT_TIMESTAMP, so text order is time order
SEVERITY = {"healthy": 0, "degraded": 1, "down": 2}
METRICS = ("cpu_pct", "memory_pct", "error_rate_pct")
MAX_OBSERVE_MINUTES = 30


def fmt(dt: datetime) -> str:
    return dt.strftime(TS_FORMAT)


def now(conn: sqlite3.Connection) -> datetime:
    return datetime.strptime(conn.execute("SELECT now FROM sim_clock").fetchone()[0], TS_FORMAT).replace(tzinfo=UTC)


def jitter(value: float, service: str, i: int) -> float:
    """Small deterministic noise, so metric history looks real but tests stay reproducible."""
    seed = sum(map(ord, service))
    return round(value * (1 + 0.03 * math.sin(i * 1.7 + seed)), 1)


def view(baselines: dict[str, dict[str, float]], effects: Iterable[dict[str, dict[str, Any]]]
         ) -> dict[str, dict[str, Any]]:
    """Services' health: baselines overlaid with the effects of the active faults (worst status wins)."""
    out = {name: {**base, "status": "healthy", "message": "OK"} for name, base in baselines.items()}
    for fault_effects in effects:
        for name, eff in fault_effects.items():
            svc = out[name]
            if SEVERITY[eff["status"]] >= SEVERITY[svc["status"]]:
                svc["status"], svc["message"] = eff["status"], eff["message"]
            for metric in METRICS:
                if metric in eff:
                    svc[metric] = max(svc[metric], eff[metric])
    return out


def _baselines(conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    rows = conn.execute("SELECT name, cpu_pct, memory_pct, error_rate_pct FROM sim_baseline").fetchall()
    return {r[0]: dict(zip(METRICS, r[1:], strict=True)) for r in rows}


def recompute(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Relapse expired masks, then write the derived health of every service. Returns it."""
    conn.execute("UPDATE sim_faults SET state='active', masked_until=NULL "
                 "WHERE state='masked' AND masked_until <= ?", (fmt(now(conn)),))
    active = [json.loads(r[0]) for r in conn.execute("SELECT effects FROM sim_faults WHERE state='active'")]
    health = view(_baselines(conn), active)
    conn.executemany(
        "UPDATE services SET status=:status, cpu_pct=:cpu_pct, memory_pct=:memory_pct, "
        "error_rate_pct=:error_rate_pct, message=:message WHERE name=:name",
        [{**h, "name": name} for name, h in health.items()],
    )
    return health


def apply_action(conn: sqlite3.Connection, action: str, target: str) -> None:
    """Apply an action (e.g. "restart", "cache") to the hidden faults, then recompute health."""
    key = f"{action}:{target}"
    for fault_id, fixed_by, masked_by, relapse in conn.execute(
            "SELECT id, fixed_by, masked_by, relapse_minutes FROM sim_faults WHERE state != 'fixed'").fetchall():
        if key in json.loads(fixed_by):
            conn.execute("UPDATE sim_faults SET state='fixed', masked_until=NULL WHERE id=?", (fault_id,))
        elif key in json.loads(masked_by):
            until = fmt(now(conn) + timedelta(minutes=relapse))
            conn.execute("UPDATE sim_faults SET state='masked', masked_until=? WHERE id=?", (until, fault_id))
    recompute(conn)


def record_metrics(conn: sqlite3.Connection, health: dict[str, dict[str, Any]]) -> None:
    ts = fmt(now(conn))
    conn.executemany(
        "INSERT INTO metrics_history (ts, service, status, cpu_pct, memory_pct, error_rate_pct) "
        "VALUES (?,?,?,?,?,?)",
        [(ts, name, h["status"], h["cpu_pct"], h["memory_pct"], h["error_rate_pct"]) for name, h in health.items()],
    )


def advance(conn: sqlite3.Connection, minutes: int) -> list[dict[str, Any]]:
    """Let `minutes` of simulated time pass, one minute at a time. Returns a snapshot per minute."""
    if not 1 <= minutes <= MAX_OBSERVE_MINUTES:
        raise ValueError(f"minutes must be between 1 and {MAX_OBSERVE_MINUTES}")
    snapshots = []
    for _ in range(minutes):
        conn.execute("UPDATE sim_clock SET now=?", (fmt(now(conn) + timedelta(minutes=1)),))
        health = recompute(conn)
        record_metrics(conn, health)
        snapshots.append({"ts": fmt(now(conn)), "status": {name: h["status"] for name, h in health.items()}})
    return snapshots


def audit(conn: sqlite3.Connection, action: str, target: str, detail: str) -> None:
    conn.execute("INSERT INTO audit_log (ts, action, target, detail) VALUES (?,?,?,?)",
                 (fmt(now(conn)), action, target, detail))
