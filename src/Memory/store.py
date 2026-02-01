import json
import sqlite3
import threading
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT,
                    embedding BLOB,
                    meta TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    text TEXT NOT NULL,
                    last_turn_id INTEGER
                )
                """
            )
            columns = [
                row["name"]
                for row in conn.execute("PRAGMA table_info(summary)").fetchall()
            ]
            if "last_turn_id" not in columns:
                conn.execute("ALTER TABLE summary ADD COLUMN last_turn_id INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_ts ON turns(ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_role ON turns(role)")
            conn.commit()

    def insert_turn(
        self,
        role: str,
        text: str,
        ts: Optional[str] = None,
        source: str = "",
        embedding: Optional[bytes] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        timestamp = ts or _utc_now()
        meta_json = json.dumps(meta) if meta else None
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO turns (ts, role, text, source, embedding, meta)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (timestamp, role, text, source, embedding, meta_json),
            )
            conn.commit()

    def insert_summary(
        self, text: str, ts: Optional[str] = None, last_turn_id: Optional[int] = None
    ) -> None:
        timestamp = ts or _utc_now()
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO summary (ts, text, last_turn_id) VALUES (?, ?, ?)",
                (timestamp, text, last_turn_id),
            )
            conn.commit()

    def latest_summary(self) -> Optional[Tuple[str, str, Optional[int]]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT ts, text, last_turn_id FROM summary ORDER BY ts DESC, id DESC LIMIT 1"
            ).fetchone()
        if not row:
            return None
        return row["ts"], row["text"], row["last_turn_id"]

    def fetch_turns_since(
        self, since_ts: Optional[str], limit: int, ascending: bool = False
    ) -> List[sqlite3.Row]:
        order = "ASC" if ascending else "DESC"
        with self._lock, self._connect() as conn:
            if since_ts:
                rows = conn.execute(
                    f"""
                    SELECT id, ts, role, text, embedding
                    FROM turns
                    WHERE ts >= ?
                    ORDER BY ts {order}, id {order}
                    LIMIT ?
                    """,
                    (since_ts, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT id, ts, role, text, embedding
                    FROM turns
                    ORDER BY ts {order}, id {order}
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return list(rows)

    def fetch_turns_after(
        self,
        since_ts: Optional[str],
        since_id: Optional[int],
        limit: int,
        ascending: bool = True,
    ) -> List[sqlite3.Row]:
        order = "ASC" if ascending else "DESC"
        with self._lock, self._connect() as conn:
            if not since_ts:
                rows = conn.execute(
                    f"""
                    SELECT id, ts, role, text, embedding
                    FROM turns
                    ORDER BY ts {order}, id {order}
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            elif since_id is None:
                rows = conn.execute(
                    f"""
                    SELECT id, ts, role, text, embedding
                    FROM turns
                    WHERE ts >= ?
                    ORDER BY ts {order}, id {order}
                    LIMIT ?
                    """,
                    (since_ts, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    f"""
                    SELECT id, ts, role, text, embedding
                    FROM turns
                    WHERE ts > ? OR (ts = ? AND id > ?)
                    ORDER BY ts {order}, id {order}
                    LIMIT ?
                    """,
                    (since_ts, since_ts, since_id, limit),
                ).fetchall()
        return list(rows)

    def count_turns_since(
        self, since_ts: Optional[str], since_id: Optional[int], role: Optional[str] = None
    ) -> int:
        with self._lock, self._connect() as conn:
            if since_ts and since_id is not None:
                if role:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM turns
                        WHERE (ts > ? OR (ts = ? AND id > ?)) AND role = ?
                        """,
                        (since_ts, since_ts, since_id, role),
                    ).fetchone()
                else:
                    row = conn.execute(
                        """
                        SELECT COUNT(*) AS count
                        FROM turns
                        WHERE ts > ? OR (ts = ? AND id > ?)
                        """,
                        (since_ts, since_ts, since_id),
                    ).fetchone()
            elif since_ts:
                if role:
                    row = conn.execute(
                        "SELECT COUNT(*) AS count FROM turns WHERE ts >= ? AND role = ?",
                        (since_ts, role),
                    ).fetchone()
                else:
                    row = conn.execute(
                        "SELECT COUNT(*) AS count FROM turns WHERE ts >= ?",
                        (since_ts,),
                    ).fetchone()
            else:
                if role:
                    row = conn.execute(
                        "SELECT COUNT(*) AS count FROM turns WHERE role = ?",
                        (role,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) AS count FROM turns").fetchone()
        return int(row["count"]) if row else 0
