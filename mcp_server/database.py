"""The agent's external environment: a small SQLite "IT service desk" database.

Visible tables: tickets, services (health + dependencies + owner), knowledge_base, audit_log, changes,
metrics_history, service_logs, pages. Hidden tables (`sim_*`): the clock, the faults and the services' healthy
baselines that drive the simulation (see simulation.py). The seed data comes from a scenario (scenarios.py).
"""

import json
import os
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp_server import simulation
from mcp_server.scenarios import DEFAULT_SCENARIO, KNOWLEDGE_BASE, SCENARIOS, Scenario

DEFAULT_DB = Path(os.environ.get("ITSM_DB_PATH", Path(__file__).resolve().parent.parent / "data" / "itsm.db"))
HIDDEN_TABLE_PREFIX = "sim_"
HISTORY_MINUTES, HISTORY_STEP = 90, 5  # seeded metric history: every 5 minutes for the last 90

SCHEMA = """
CREATE TABLE tickets (
    id TEXT PRIMARY KEY, title TEXT, description TEXT, requester TEXT,
    priority TEXT, status TEXT, service TEXT, notes TEXT DEFAULT '', parent_id TEXT
);
CREATE TABLE services (
    name TEXT PRIMARY KEY, status TEXT, cpu_pct REAL, memory_pct REAL,
    error_rate_pct REAL, depends_on TEXT, message TEXT, owner TEXT, kind TEXT
);
CREATE TABLE knowledge_base (id TEXT PRIMARY KEY, title TEXT, content TEXT);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT DEFAULT CURRENT_TIMESTAMP,
    action TEXT, target TEXT, detail TEXT
);
CREATE TABLE changes (
    id TEXT PRIMARY KEY, ts TEXT, service TEXT, kind TEXT, summary TEXT, author TEXT, status TEXT
);
CREATE TABLE metrics_history (
    ts TEXT, service TEXT, status TEXT, cpu_pct REAL, memory_pct REAL, error_rate_pct REAL
);
CREATE TABLE service_logs (ts TEXT, service TEXT, level TEXT, message TEXT);
CREATE TABLE pages (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, team TEXT, severity TEXT, message TEXT);

CREATE TABLE sim_clock (now TEXT);
CREATE TABLE sim_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE sim_baseline (name TEXT PRIMARY KEY, cpu_pct REAL, memory_pct REAL, error_rate_pct REAL);
CREATE TABLE sim_faults (
    id TEXT PRIMARY KEY, fixed_by TEXT, masked_by TEXT, relapse_minutes INTEGER,
    state TEXT, masked_until TEXT, effects TEXT
);
CREATE INDEX metrics_by_service ON metrics_history (service, ts);
"""


def connect(db_path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open the database with dict-like rows."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def reset_database(db_path: Path = DEFAULT_DB, scenario: str | None = None) -> Path:
    """(Re)create the database with a scenario's demo data (default: ITSM_SCENARIO, else cache-outage)."""
    name = scenario or os.getenv("ITSM_SCENARIO", DEFAULT_SCENARIO)
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{name}'. Known: {sorted(SCENARIOS)}")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _seed(conn, SCENARIOS[name], datetime.now(UTC).replace(microsecond=0))
    conn.close()
    return db_path


def current_scenario(db_path: Path = DEFAULT_DB) -> str | None:
    """The scenario the database was seeded with (None if it does not exist or predates scenarios)."""
    if not db_path.exists():
        return None
    with connect(db_path) as conn:
        try:
            row = conn.execute("SELECT value FROM sim_meta WHERE key = 'scenario'").fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def _seed(conn: sqlite3.Connection, sc: Scenario, start: datetime) -> None:
    def ago(minutes: int) -> str:
        return simulation.fmt(start - timedelta(minutes=minutes))

    conn.execute("INSERT INTO sim_clock VALUES (?)", (simulation.fmt(start),))
    conn.execute("INSERT INTO sim_meta VALUES ('scenario', ?)", (sc.name,))
    conn.executemany("INSERT INTO sim_baseline VALUES (?,?,?,?)",
                     [(s.name, s.cpu_pct, s.memory_pct, s.error_rate_pct) for s in sc.services])
    conn.executemany(
        "INSERT INTO services (name, depends_on, owner, kind) VALUES (?,?,?,?)",
        [(s.name, ",".join(s.depends_on) or None, s.owner, s.kind) for s in sc.services],
    )
    conn.executemany(
        "INSERT INTO sim_faults VALUES (?,?,?,?, 'active', NULL, ?)",
        [(f.id, json.dumps(f.fixed_by), json.dumps(f.masked_by), f.relapse_minutes, json.dumps(f.effects))
         for f in sc.faults],
    )
    conn.executemany("INSERT INTO tickets (id, title, description, requester, priority, status, service) "
                     "VALUES (?,?,?,?,?,?,?)", sc.tickets)
    conn.executemany("INSERT INTO knowledge_base VALUES (?,?,?)", KNOWLEDGE_BASE + sc.extra_kb)
    conn.executemany("INSERT INTO changes VALUES (?,?,?,?,?,?, 'applied')",
                     [(cid, ago(m), svc, kind, summary, author) for cid, m, svc, kind, summary, author in sc.changes])
    conn.executemany("INSERT INTO service_logs VALUES (?,?,?,?)",
                     [(ago(m), svc, level, msg) for m, svc, level, msg in sc.logs])
    _seed_metrics(conn, sc, ago)
    simulation.recompute(conn)


def _seed_metrics(conn: sqlite3.Connection, sc: Scenario, ago: Callable[[int], str]) -> None:
    """Metric history consistent with the faults: baseline before each fault started, symptoms after."""
    baselines = {s.name: {"cpu_pct": s.cpu_pct, "memory_pct": s.memory_pct, "error_rate_pct": s.error_rate_pct}
                 for s in sc.services}
    rows = []
    for i, minutes in enumerate(range(HISTORY_MINUTES, 0, -HISTORY_STEP)):
        active = [f.effects for f in sc.faults if f.started_minutes_ago >= minutes]
        for name, h in simulation.view(baselines, active).items():
            rows.append((ago(minutes), name, h["status"],
                         *(simulation.jitter(h[m], name, i) for m in simulation.METRICS)))
    conn.executemany("INSERT INTO metrics_history VALUES (?,?,?,?,?,?)", rows)
