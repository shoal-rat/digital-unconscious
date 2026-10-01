"""The memory: one SQLite file.

Raw traces are the only high-resolution personal data and they expire after
``sense.retention_days``. Everything derived from them (threads, dreams,
sparks) is abstract enough to keep, which mirrors the idea behind the app:
details fade, themes remain.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS traces(
  id INTEGER PRIMARY KEY,
  day TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL,
  seconds REAL NOT NULL DEFAULT 0,
  kind TEXT NOT NULL,
  source TEXT NOT NULL,
  app TEXT NOT NULL DEFAULT '',
  category TEXT NOT NULL DEFAULT '',
  title TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL DEFAULT '',
  domain TEXT NOT NULL DEFAULT '',
  body TEXT NOT NULL DEFAULT '',
  subject_key TEXT NOT NULL,
  subject TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS traces_day ON traces(day, subject_key);

CREATE TABLE IF NOT EXISTS threads(
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  gist TEXT NOT NULL DEFAULT '',
  keywords TEXT NOT NULL DEFAULT '[]',
  hue INTEGER NOT NULL DEFAULT 0,
  state TEXT NOT NULL DEFAULT 'active',
  merged_into INTEGER,
  first_day TEXT,
  last_day TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS thread_days(
  thread_id INTEGER NOT NULL,
  day TEXT NOT NULL,
  seconds REAL NOT NULL DEFAULT 0,
  visits INTEGER NOT NULL DEFAULT 0,
  subjects TEXT NOT NULL DEFAULT '[]',
  note TEXT NOT NULL DEFAULT '',
  PRIMARY KEY(thread_id, day)
);
CREATE INDEX IF NOT EXISTS thread_days_day ON thread_days(day);

CREATE TABLE IF NOT EXISTS subject_threads(
  day TEXT NOT NULL,
  subject_key TEXT NOT NULL,
  thread_id INTEGER NOT NULL,
  PRIMARY KEY(day, subject_key)
);

CREATE TABLE IF NOT EXISTS digests(
  day TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  trace_count INTEGER NOT NULL,
  payload TEXT NOT NULL,
  model TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS dreams(
  id INTEGER PRIMARY KEY,
  day TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  reflection TEXT NOT NULL DEFAULT '',
  undercurrent TEXT NOT NULL DEFAULT '',
  payload TEXT NOT NULL DEFAULT '{}',
  models TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sparks(
  id INTEGER PRIMARY KEY,
  dream_id INTEGER,
  day TEXT NOT NULL,
  title TEXT NOT NULL,
  mechanism TEXT NOT NULL,
  question TEXT NOT NULL DEFAULT '',
  insight TEXT NOT NULL DEFAULT '',
  first_step TEXT NOT NULL DEFAULT '',
  kill TEXT NOT NULL DEFAULT '',
  field TEXT NOT NULL DEFAULT '',
  thread_ids TEXT NOT NULL DEFAULT '[]',
  evidence TEXT NOT NULL DEFAULT '[]',
  search_terms TEXT NOT NULL DEFAULT '[]',
  scores TEXT NOT NULL DEFAULT '{}',
  score REAL NOT NULL DEFAULT 0,
  objection TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'new',
  reason TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sparks_day ON sparks(day);

CREATE TABLE IF NOT EXISTS dives(
  id INTEGER PRIMARY KEY,
  spark_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  report TEXT NOT NULL DEFAULT '{}',
  papers TEXT NOT NULL DEFAULT '[]',
  model TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  ref TEXT NOT NULL DEFAULT '',
  data TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_kind ON events(kind, ts);

CREATE TABLE IF NOT EXISTS llm_calls(
  id INTEGER PRIMARY KEY,
  ts TEXT NOT NULL,
  role TEXT NOT NULL,
  provider TEXT NOT NULL,
  model TEXT NOT NULL,
  ok INTEGER NOT NULL,
  ms INTEGER NOT NULL,
  tokens_in INTEGER NOT NULL DEFAULT 0,
  tokens_out INTEGER NOT NULL DEFAULT 0,
  cost REAL NOT NULL DEFAULT 0,
  error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  ref TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL,
  step TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  finished_at TEXT,
  result TEXT NOT NULL DEFAULT '{}',
  error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS kv(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

JSON_FIELDS = {
    "keywords", "subjects", "payload", "models", "thread_ids", "evidence", "search_terms",
    "scores", "report", "papers", "data", "result",
}
PALETTE_SIZE = 10
SPARK_STATUSES = {"new", "kept", "pursuing", "dismissed", "done"}
THREAD_STATES = {"active", "pinned", "muted", "merged"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def today() -> str:
    return datetime.now().astimezone().date().isoformat()


def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_FIELDS & out.keys():
        try:
            out[key] = json.loads(out[key]) if out[key] else None
        except (TypeError, json.JSONDecodeError):
            pass
    return out


def _rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [_row(r) for r in rows]  # type: ignore[misc]


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Store:
    """Thin repository over SQLite. Every method opens a short-lived connection,
    which keeps the sensor thread, the web server and CLI processes independent."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self.tx() as db:
            db.executescript(SCHEMA)
            db.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO NOTHING",
                (str(SCHEMA_VERSION),),
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock, self.connect() as db:
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    # ------------------------------------------------------------------ kv

    def get(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            row = db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def put(self, key: str, value: Any) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, _dump(value)),
            )

    # -------------------------------------------------------------- traces

    def add_trace(self, **fields: Any) -> int:
        fields.setdefault("ended_at", fields["started_at"])
        fields.setdefault("seconds", 0.0)
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.tx() as db:
            cur = db.execute(f"INSERT INTO traces({columns}) VALUES({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def extend_trace(self, trace_id: int, ended_at: str, seconds: float) -> None:
        with self.tx() as db:
            db.execute("UPDATE traces SET ended_at=?, seconds=? WHERE id=?", (ended_at, seconds, trace_id))

    def traces(self, day: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return _rows(db.execute("SELECT * FROM traces WHERE day=? ORDER BY started_at", (day,)).fetchall())

    def trace_count(self, day: str | None = None) -> int:
        with self.connect() as db:
            if day:
                return db.execute("SELECT COUNT(*) FROM traces WHERE day=?", (day,)).fetchone()[0]
            return db.execute("SELECT COUNT(*) FROM traces").fetchone()[0]

    def day_seconds(self, day: str) -> float:
        with self.connect() as db:
            value = db.execute("SELECT COALESCE(SUM(seconds),0) FROM traces WHERE day=?", (day,)).fetchone()[0]
        return float(value or 0)

    def subjects(self, day: str) -> list[dict[str, Any]]:
        """Aggregate one day's traces into subjects (what was attended to)."""
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT subject_key, MIN(subject) AS subject, MIN(kind) AS kind, MIN(category) AS category,
                       MIN(app) AS app, MIN(domain) AS domain, MAX(url) AS url,
                       SUM(seconds) AS seconds, COUNT(*) AS visits,
                       MIN(started_at) AS first_at, MAX(ended_at) AS last_at,
                       MAX(body) AS body
                FROM traces WHERE day=? GROUP BY subject_key
                ORDER BY seconds DESC, visits DESC
                """,
                (day,),
            ).fetchall()
        return _rows(rows)

    def days_with_traces(self, limit: int = 60) -> list[str]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT DISTINCT day FROM traces ORDER BY day DESC LIMIT ?", (limit,)
            ).fetchall()
        return [r["day"] for r in rows]

    def delete_traces(self, day: str, source: str) -> int:
        with self.tx() as db:
            return db.execute("DELETE FROM traces WHERE day=? AND source=?", (day, source)).rowcount

    def forget_day(self, day: str) -> int:
        with self.tx() as db:
            n = db.execute("DELETE FROM traces WHERE day=?", (day,)).rowcount
            db.execute("DELETE FROM subject_threads WHERE day=?", (day,))
            db.execute("DELETE FROM digests WHERE day=?", (day,))
            db.execute("DELETE FROM thread_days WHERE day=?", (day,))
        return n

    def forget_before(self, day: str) -> int:
        """Expire raw traces older than ``day``; abstractions are kept."""
        with self.tx() as db:
            n = db.execute("DELETE FROM traces WHERE day<?", (day,)).rowcount
            db.execute("DELETE FROM subject_threads WHERE day<?", (day,))
        return n

    def forget_everything(self) -> None:
        with self.tx() as db:
            for table in (
                "traces", "threads", "thread_days", "subject_threads", "digests", "dreams",
                "sparks", "dives", "events", "llm_calls", "jobs",
            ):
                db.execute(f"DELETE FROM {table}")

    # ------------------------------------------------------------- threads

    def threads(self, states: set[str] | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = _rows(db.execute("SELECT * FROM threads ORDER BY last_day DESC, id").fetchall())
        if states:
            rows = [r for r in rows if r["state"] in states]
        return rows

    def thread(self, thread_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM threads WHERE id=?", (thread_id,)).fetchone())

    def create_thread(self, name: str, gist: str, keywords: list[str], day: str) -> int:
        with self.tx() as db:
            count = db.execute("SELECT COUNT(*) FROM threads").fetchone()[0]
            cur = db.execute(
                "INSERT INTO threads(name, gist, keywords, hue, state, first_day, last_day, created_at) "
                "VALUES(?, ?, ?, ?, 'active', ?, ?, ?)",
                (name, gist, _dump(keywords), count % PALETTE_SIZE, day, day, now_iso()),
            )
            return int(cur.lastrowid)

    def update_thread(self, thread_id: int, **fields: Any) -> None:
        allowed = {"name", "gist", "keywords", "state", "merged_into", "first_day", "last_day", "hue"}
        sets = {k: (_dump(v) if k == "keywords" else v) for k, v in fields.items() if k in allowed}
        if "state" in sets and sets["state"] not in THREAD_STATES:
            raise ValueError(f"unknown thread state {sets['state']!r}")
        if not sets:
            return
        assignments = ", ".join(f"{k}=?" for k in sets)
        with self.tx() as db:
            db.execute(f"UPDATE threads SET {assignments} WHERE id=?", (*sets.values(), thread_id))

    def merge_threads(self, source_id: int, target_id: int) -> None:
        if source_id == target_id:
            return
        with self.tx() as db:
            for row in db.execute(
                "SELECT day, seconds, visits, subjects FROM thread_days WHERE thread_id=?", (source_id,)
            ).fetchall():
                existing = db.execute(
                    "SELECT seconds, visits, subjects FROM thread_days WHERE thread_id=? AND day=?",
                    (target_id, row["day"]),
                ).fetchone()
                if existing:
                    subjects = json.loads(existing["subjects"]) + json.loads(row["subjects"])
                    db.execute(
                        "UPDATE thread_days SET seconds=?, visits=?, subjects=? WHERE thread_id=? AND day=?",
                        (
                            existing["seconds"] + row["seconds"],
                            existing["visits"] + row["visits"],
                            _dump(list(dict.fromkeys(subjects))[:12]),
                            target_id,
                            row["day"],
                        ),
                    )
                else:
                    db.execute(
                        "INSERT INTO thread_days(thread_id, day, seconds, visits, subjects) VALUES(?,?,?,?,?)",
                        (target_id, row["day"], row["seconds"], row["visits"], row["subjects"]),
                    )
            db.execute("DELETE FROM thread_days WHERE thread_id=?", (source_id,))
            db.execute("UPDATE subject_threads SET thread_id=? WHERE thread_id=?", (target_id, source_id))
            db.execute(
                "UPDATE threads SET state='merged', merged_into=? WHERE id=?", (target_id, source_id)
            )
            span = db.execute(
                "SELECT MIN(day), MAX(day) FROM thread_days WHERE thread_id=?", (target_id,)
            ).fetchone()
            db.execute(
                "UPDATE threads SET first_day=COALESCE(?, first_day), last_day=COALESCE(?, last_day) WHERE id=?",
                (span[0], span[1], target_id),
            )

    def replace_day_assignments(
        self,
        day: str,
        thread_days: list[dict[str, Any]],
        subject_map: dict[str, int],
    ) -> None:
        """Atomically replace everything a digest derived for one day."""
        with self.tx() as db:
            db.execute("DELETE FROM thread_days WHERE day=?", (day,))
            db.execute("DELETE FROM subject_threads WHERE day=?", (day,))
            for item in thread_days:
                db.execute(
                    "INSERT INTO thread_days(thread_id, day, seconds, visits, subjects, note) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(thread_id, day) DO UPDATE SET seconds=seconds+excluded.seconds, "
                    "visits=visits+excluded.visits, subjects=excluded.subjects, note=excluded.note",
                    (
                        item["thread_id"], day, item["seconds"], item["visits"],
                        _dump(item.get("subjects", [])[:12]), item.get("note", ""),
                    ),
                )
                db.execute(
                    "UPDATE threads SET last_day=MAX(COALESCE(last_day, ?), ?), "
                    "first_day=MIN(COALESCE(first_day, ?), ?) WHERE id=?",
                    (day, day, day, day, item["thread_id"]),
                )
            db.executemany(
                "INSERT OR REPLACE INTO subject_threads(day, subject_key, thread_id) VALUES(?,?,?)",
                [(day, key, tid) for key, tid in subject_map.items()],
            )

    def thread_days(self, since: str, thread_id: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM thread_days WHERE day>=?"
        args: list[Any] = [since]
        if thread_id is not None:
            query += " AND thread_id=?"
            args.append(thread_id)
        with self.connect() as db:
            return _rows(db.execute(query + " ORDER BY day", args).fetchall())

    def subject_threads(self, day: str) -> dict[str, int]:
        with self.connect() as db:
            rows = db.execute("SELECT subject_key, thread_id FROM subject_threads WHERE day=?", (day,)).fetchall()
        return {r["subject_key"]: r["thread_id"] for r in rows}

    # ------------------------------------------------------------- digests

    def save_digest(self, day: str, payload: dict[str, Any], trace_count: int, model: str) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO digests(day, created_at, trace_count, payload, model) VALUES(?,?,?,?,?)",
                (day, now_iso(), trace_count, _dump(payload), model),
            )

    def digest(self, day: str) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM digests WHERE day=?", (day,)).fetchone())

    # -------------------------------------------------------------- dreams

    def save_dream(
        self,
        day: str,
        *,
        title: str,
        reflection: str,
        undercurrent: str,
        payload: dict[str, Any],
        models: dict[str, Any],
    ) -> int:
        """Store (or replace) a day's dream. Unreviewed sparks from an earlier
        dream of the same day are discarded; sparks the user acted on survive."""
        with self.tx() as db:
            old = db.execute("SELECT id FROM dreams WHERE day=?", (day,)).fetchone()
            if old:
                db.execute("DELETE FROM sparks WHERE dream_id=? AND status='new'", (old["id"],))
                db.execute(
                    "UPDATE dreams SET created_at=?, title=?, reflection=?, undercurrent=?, payload=?, models=? "
                    "WHERE id=?",
                    (now_iso(), title, reflection, undercurrent, _dump(payload), _dump(models), old["id"]),
                )
                return int(old["id"])
            cur = db.execute(
                "INSERT INTO dreams(day, created_at, title, reflection, undercurrent, payload, models) "
                "VALUES(?,?,?,?,?,?,?)",
                (day, now_iso(), title, reflection, undercurrent, _dump(payload), _dump(models)),
            )
            return int(cur.lastrowid)

    def dream(self, day: str) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM dreams WHERE day=?", (day,)).fetchone())

    def latest_dream(self) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM dreams ORDER BY day DESC LIMIT 1").fetchone())

    def dreams(self, limit: int = 60) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM sparks s WHERE s.dream_id=d.id) AS spark_count "
                "FROM dreams d ORDER BY day DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return _rows(rows)

    def dream_number(self, day: str) -> int:
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM dreams WHERE day<=?", (day,)).fetchone()[0]

    # -------------------------------------------------------------- sparks

    def add_spark(self, **fields: Any) -> int:
        stamp = now_iso()
        fields.setdefault("created_at", stamp)
        fields.setdefault("updated_at", stamp)
        for key in ("thread_ids", "evidence", "search_terms", "scores"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = _dump(fields[key])
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.tx() as db:
            cur = db.execute(f"INSERT INTO sparks({columns}) VALUES({marks})", tuple(fields.values()))
            return int(cur.lastrowid)

    def spark(self, spark_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM sparks WHERE id=?", (spark_id,)).fetchone())

    def sparks(
        self,
        *,
        status: str | None = None,
        day: str | None = None,
        dream_id: int | None = None,
        thread_id: int | None = None,
        query: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses, args = [], []
        if status:
            clauses.append("status=?")
            args.append(status)
        if day:
            clauses.append("day=?")
            args.append(day)
        if dream_id is not None:
            clauses.append("dream_id=?")
            args.append(dream_id)
        if query:
            clauses.append("(title LIKE ? OR question LIKE ? OR insight LIKE ?)")
            args.extend([f"%{query}%"] * 3)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.connect() as db:
            rows = _rows(
                db.execute(
                    f"SELECT * FROM sparks {where} ORDER BY day DESC, score DESC LIMIT ?", (*args, limit)
                ).fetchall()
            )
        if thread_id is not None:
            rows = [r for r in rows if thread_id in (r.get("thread_ids") or [])]
        return rows

    def set_spark_status(self, spark_id: int, status: str, reason: str = "") -> None:
        if status not in SPARK_STATUSES:
            raise ValueError(f"unknown spark status {status!r}")
        with self.tx() as db:
            db.execute(
                "UPDATE sparks SET status=?, reason=?, updated_at=? WHERE id=?",
                (status, reason, now_iso(), spark_id),
            )

    # --------------------------------------------------------------- dives

    def add_dive(self, spark_id: int, report: dict[str, Any], papers: list[dict[str, Any]], model: str) -> int:
        with self.tx() as db:
            cur = db.execute(
                "INSERT INTO dives(spark_id, created_at, report, papers, model) VALUES(?,?,?,?,?)",
                (spark_id, now_iso(), _dump(report), _dump(papers), model),
            )
            return int(cur.lastrowid)

    def dives(self, spark_id: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            return _rows(
                db.execute("SELECT * FROM dives WHERE spark_id=? ORDER BY id DESC", (spark_id,)).fetchall()
            )

    # -------------------------------------------------------------- events

    def log_event(self, kind: str, ref: str = "", data: dict[str, Any] | None = None) -> None:
        with self.tx() as db:
            db.execute(
                "INSERT INTO events(ts, kind, ref, data) VALUES(?,?,?,?)",
                (now_iso(), kind, ref, _dump(data or {})),
            )

    def events(self, kind: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as db:
            return _rows(
                db.execute(
                    "SELECT * FROM events WHERE kind=? ORDER BY id DESC LIMIT ?", (kind, limit)
                ).fetchall()
            )

    # ----------------------------------------------------------- llm usage

    def log_llm(self, **fields: Any) -> None:
        fields.setdefault("ts", now_iso())
        columns = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self.tx() as db:
            db.execute(f"INSERT INTO llm_calls({columns}) VALUES({marks})", tuple(fields.values()))

    def llm_usage(self, since: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT role, provider, model, COUNT(*) AS calls, SUM(ok) AS ok, "
                "SUM(tokens_in) AS tokens_in, SUM(tokens_out) AS tokens_out, SUM(cost) AS cost, "
                "AVG(ms) AS avg_ms FROM llm_calls WHERE ts>=? GROUP BY role, provider, model "
                "ORDER BY calls DESC",
                (since,),
            ).fetchall()
        return _rows(rows)

    def last_llm_error(self) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(
                db.execute("SELECT * FROM llm_calls WHERE ok=0 ORDER BY id DESC LIMIT 1").fetchone()
            )

    # ---------------------------------------------------------------- jobs

    def create_job(self, kind: str, ref: str = "") -> str:
        job_id = uuid.uuid4().hex[:12]
        with self.tx() as db:
            db.execute(
                "INSERT INTO jobs(id, kind, ref, state, created_at) VALUES(?,?,?,'queued',?)",
                (job_id, kind, ref, now_iso()),
            )
        return job_id

    def update_job(self, job_id: str, **fields: Any) -> None:
        if "result" in fields and not isinstance(fields["result"], str):
            fields["result"] = _dump(fields["result"])
        if fields.get("state") in {"done", "failed"}:
            fields.setdefault("finished_at", now_iso())
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self.tx() as db:
            db.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*fields.values(), job_id))

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            return _row(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def active_jobs(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return _rows(
                db.execute(
                    "SELECT * FROM jobs WHERE state IN ('queued','running') ORDER BY created_at"
                ).fetchall()
            )

    def fail_stale_jobs(self) -> None:
        """Jobs still marked running when a process starts were orphaned by a crash."""
        with self.tx() as db:
            db.execute(
                "UPDATE jobs SET state='failed', error='interrupted', finished_at=? "
                "WHERE state IN ('queued','running')",
                (now_iso(),),
            )
