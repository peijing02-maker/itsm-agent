"""The agent's external environment: a small SQLite "IT service desk" database.

Visible tables: tickets, services (health + dependencies + owner), knowledge_base, audit_log, changes,
metrics_history, service_logs, pages. Hidden tables (`sim_*`): the clock, the faults and the services' healthy
baselines that drive the simulation (see simulation.py). The seed data comes from the demo world (world.py).
"""

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp_server import simulation
from mcp_server.world import DEMO_WORLD, KNOWLEDGE_BASE, World

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


@contextmanager
def connect(db_path: Path = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    """Open the database with dict-like rows; commit (or roll back) and close on exit.

    sqlite3's own context manager never closes the connection, and on Windows an open handle stops the file
    from being replaced.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def reset_database(db_path: Path = DEFAULT_DB, world: World = DEMO_WORLD) -> Path:
    """(Re)create the database with a world's seed data. Everything the agent changed is discarded."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        # Drop and recreate in place rather than deleting the file: Windows refuses to delete a file that another
        # process (the MCP server, a second browser tab) still has open.
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                             "AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            conn.execute(f'DROP TABLE "{table}"')
        conn.executescript(SCHEMA)
        _seed(conn, world, datetime.now(UTC).replace(microsecond=0))
    return db_path


def seeded_world(db_path: Path = DEFAULT_DB) -> str | None:
    """The name of the world the database was seeded with (None if it does not exist or predates worlds)."""
    if not db_path.exists():
        return None
    with connect(db_path) as conn:
        try:
            row = conn.execute("SELECT value FROM sim_meta WHERE key = 'world'").fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def _seed(conn: sqlite3.Connection, world: World, start: datetime) -> None:
    def ago(minutes: int) -> str:
        return simulation.fmt(start - timedelta(minutes=minutes))

    conn.execute("INSERT INTO sim_clock VALUES (?)", (simulation.fmt(start),))
    conn.execute("INSERT INTO sim_meta VALUES ('world', ?)", (world.name,))
    conn.executemany("INSERT INTO sim_baseline VALUES (?,?,?,?)",
                     [(s.name, s.cpu_pct, s.memory_pct, s.error_rate_pct) for s in world.services])
    conn.executemany(
        "INSERT INTO services (name, depends_on, owner, kind) VALUES (?,?,?,?)",
        [(s.name, ",".join(s.depends_on) or None, s.owner, s.kind) for s in world.services],
    )
    conn.executemany(
        "INSERT INTO sim_faults VALUES (?,?,?,?, 'active', NULL, ?)",
        [(f.id, json.dumps(f.fixed_by), json.dumps(f.masked_by), f.relapse_minutes, json.dumps(f.effects))
         for f in world.faults],
    )
    conn.executemany("INSERT INTO tickets (id, title, description, requester, priority, status, service) "
                     "VALUES (?,?,?,?,?,?,?)", world.tickets)
    conn.executemany("INSERT INTO knowledge_base VALUES (?,?,?)", KNOWLEDGE_BASE + world.extra_kb)
    conn.executemany("INSERT INTO changes VALUES (?,?,?,?,?,?, 'applied')",
                     [(cid, ago(m), svc, kind, summary, author) for cid, m, svc, kind, summary, author in world.changes])
    conn.executemany("INSERT INTO service_logs VALUES (?,?,?,?)",
                     [(ago(m), svc, level, msg) for m, svc, level, msg in world.logs])
    _seed_metrics(conn, world, ago)
    simulation.recompute(conn)


def _seed_metrics(conn: sqlite3.Connection, world: World, ago: Callable[[int], str]) -> None:
    """Metric history consistent with the faults: baseline before each fault started, symptoms after."""
    baselines = {s.name: {"cpu_pct": s.cpu_pct, "memory_pct": s.memory_pct, "error_rate_pct": s.error_rate_pct}
                 for s in world.services}
    rows = []
    for i, minutes in enumerate(range(HISTORY_MINUTES, 0, -HISTORY_STEP)):
        active = [f.effects for f in world.faults if f.started_minutes_ago >= minutes]
        for name, h in simulation.view(baselines, active).items():
            rows.append((ago(minutes), name, h["status"],
                         *(simulation.jitter(h[m], name, i) for m in simulation.METRICS)))
    conn.executemany("INSERT INTO metrics_history VALUES (?,?,?,?,?,?)", rows)
