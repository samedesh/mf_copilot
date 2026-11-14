"""Persistent per-session memory (SQLite) so follow-ups like 'what about the regular plan?' work."""
from __future__ import annotations
import sqlite3
import time
from contextlib import closing

from app.config import settings


def _conn() -> sqlite3.Connection:
    settings.session_db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(settings.session_db)
    c.execute("CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY AUTOINCREMENT, "
              "session_id TEXT, role TEXT, content TEXT, ts REAL)")
    return c


def add_turn(session_id: str, role: str, content: str) -> None:
    with closing(_conn()) as c:
        c.execute("INSERT INTO turns(session_id, role, content, ts) VALUES (?,?,?,?)",
                  (session_id, role, content, time.time()))
        c.commit()


def history(session_id: str, n: int = 6) -> list[dict]:
    with closing(_conn()) as c:
        rows = c.execute("SELECT role, content FROM turns WHERE session_id=? ORDER BY id DESC LIMIT ?",
                         (session_id, n)).fetchall()
    return [{"role": r, "content": t} for r, t in reversed(rows)]
