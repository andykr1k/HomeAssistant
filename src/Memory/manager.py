import logging
import os
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .embedder import Embedder
from .store import MemoryStore


@dataclass
class MemoryHit:
    ts: str
    role: str
    text: str
    score: float


class MemoryManager:
    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)

        self._enabled = self._env_flag("MEMORY_ENABLED", True)
        self._db_path = os.getenv("MEMORY_DB_PATH", "memory.db")
        self._model_name = os.getenv("MEMORY_EMBED_MODEL", "all-MiniLM-L6-v2")
        self._embed_device = os.getenv("MEMORY_EMBED_DEVICE", "cpu")
        self._top_k = int(os.getenv("MEMORY_TOP_K", "4"))
        self._max_chars = int(os.getenv("MEMORY_MAX_CHARS", "1500"))
        self._recency_days = int(os.getenv("MEMORY_RECENCY_DAYS", "45"))
        self._candidate_limit = int(os.getenv("MEMORY_CANDIDATE_LIMIT", "200"))

        self._summary_enabled = self._env_flag("MEMORY_SUMMARY_ENABLED", True)
        self._summary_every = int(os.getenv("MEMORY_SUMMARY_EVERY", "10"))
        self._summary_turn_limit = int(os.getenv("MEMORY_SUMMARY_TURN_LIMIT", "40"))
        self._summary_turn_chars = int(os.getenv("MEMORY_SUMMARY_TURN_CHARS", "300"))
        self._summary_max_chars = int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "1200"))

        self._store: Optional[MemoryStore] = None
        self._embedder: Optional[Embedder] = None
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._summarizer: Optional[Callable[[str, str, int], str]] = None
        self._summary_pending = False
        self._summary_lock = threading.Lock()

        if not self._enabled:
            if self._debug:
                self._logger.debug("Memory disabled via MEMORY_ENABLED.")
            return

        if self._debug:
            self._logger.debug(
                "Memory config db=%s model=%s device=%s top_k=%s max_chars=%s recency_days=%s",
                self._db_path,
                self._model_name,
                self._embed_device,
                self._top_k,
                self._max_chars,
                self._recency_days,
            )
            self._logger.debug(
                "Memory summary config enabled=%s every=%s turn_limit=%s turn_chars=%s max_chars=%s",
                self._summary_enabled,
                self._summary_every,
                self._summary_turn_limit,
                self._summary_turn_chars,
                self._summary_max_chars,
            )

        self._store = MemoryStore(self._db_path)
        self._embedder = self._init_embedder()

        self._queue = queue.Queue()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def enabled(self) -> bool:
        return bool(self._enabled and self._store)

    def set_summarizer(self, summarizer: Optional[Callable[[str, str, int], str]]) -> None:
        self._summarizer = summarizer
        if self._debug and summarizer:
            self._logger.debug("Memory summarizer enabled.")
        self._maybe_queue_summary()

    def record_turn(self, user_text: str, assistant_text: Optional[str], source: str = "") -> None:
        if not self.enabled or not self._queue:
            return
        if not user_text and not assistant_text:
            return
        if self._debug:
            self._logger.debug(
                "Memory queue turn source=%s user_len=%s assistant_len=%s",
                source,
                len(user_text or ""),
                len(assistant_text or ""),
            )
        payload = {
            "type": "turn",
            "ts": datetime.utcnow().isoformat(timespec="seconds"),
            "user_text": user_text or "",
            "assistant_text": assistant_text or "",
            "source": source,
        }
        self._queue.put(payload)

    def build_context(self, query_text: str, max_chars: Optional[int] = None) -> str:
        if not self.enabled:
            return ""
        if not self._store:
            return ""

        summary_text = ""
        summary_entry = self._store.latest_summary()
        if summary_entry:
            summary_text = summary_entry[1]

        memories = self._retrieve_memories(query_text)
        budget = max_chars if max_chars is not None else self._max_chars
        context = self._format_context(summary_text, memories, budget)
        if self._debug:
            self._logger.debug(
                "Memory context built summary_len=%s memories=%s context_len=%s",
                len(summary_text or ""),
                len(memories),
                len(context or ""),
            )
        return context

    def _init_embedder(self) -> Optional[Embedder]:
        try:
            return Embedder(self._model_name, device=self._embed_device, normalize=True)
        except Exception as exc:
            if self._debug:
                self._logger.debug("Memory embeddings disabled: %s", exc)
            return None

    def _worker_loop(self) -> None:
        if not self._queue:
            return
        while True:
            task = self._queue.get()
            if task is None:
                break
            task_type = task.get("type")
            if self._debug:
                self._logger.debug("Memory worker task=%s", task_type)
            if task_type == "turn":
                self._handle_turn_task(task)
            elif task_type == "summary":
                self._handle_summary_task()

    def _handle_turn_task(self, task: Dict[str, str]) -> None:
        if not self._store:
            return
        ts = task.get("ts") or datetime.utcnow().isoformat(timespec="seconds")
        source = task.get("source", "")

        user_text = task.get("user_text", "").strip()
        if user_text:
            embedding = self._embed_text(user_text)
            self._store.insert_turn(
                role="user",
                text=user_text,
                ts=ts,
                source=source,
                embedding=embedding,
            )

        assistant_text = task.get("assistant_text", "").strip()
        if assistant_text:
            embedding = self._embed_text(assistant_text)
            self._store.insert_turn(
                role="assistant",
                text=assistant_text,
                ts=ts,
                source=source,
                embedding=embedding,
            )

        self._maybe_queue_summary()

    def _embed_text(self, text: str) -> Optional[bytes]:
        if not self._embedder or not text:
            return None
        vector = self._embedder.embed(text)
        if vector is None or vector.size == 0:
            return None
        return vector.tobytes()

    def _maybe_queue_summary(self) -> None:
        if not self._summary_enabled or not self._summarizer or not self._store:
            return
        with self._summary_lock:
            if self._summary_pending:
                return
            summary_entry = self._store.latest_summary()
            since_ts = summary_entry[0] if summary_entry else None
            since_id = summary_entry[2] if summary_entry else None
            turn_count = self._store.count_turns_since(since_ts, since_id, role="user")
            if self._debug:
                self._logger.debug(
                    "Memory summary check since_ts=%s since_id=%s user_turns=%s",
                    since_ts,
                    since_id,
                    turn_count,
                )
            if turn_count < self._summary_every:
                return
            if self._queue:
                self._summary_pending = True
                self._queue.put({"type": "summary"})

    def _handle_summary_task(self) -> None:
        with self._summary_lock:
            self._summary_pending = False
        if not self._summarizer or not self._store:
            return

        summary_entry = self._store.latest_summary()
        previous_summary = summary_entry[1] if summary_entry else ""
        since_ts = summary_entry[0] if summary_entry else None
        since_id = summary_entry[2] if summary_entry else None
        turns = self._store.fetch_turns_after(
            since_ts,
            since_id,
            limit=self._summary_turn_limit,
            ascending=True,
        )
        if not turns:
            return

        turns_text = self._format_turns_for_summary(turns)
        if not turns_text:
            return

        try:
            updated = self._summarizer(previous_summary, turns_text, self._summary_max_chars)
        except Exception as exc:
            self._logger.warning("Memory summarizer failed: %s", exc)
            return

        updated = (updated or "").strip()
        if not updated:
            return

        if len(updated) > self._summary_max_chars:
            updated = updated[: max(0, self._summary_max_chars - 3)] + "..."

        last_turn = turns[-1]
        self._store.insert_summary(
            updated,
            ts=last_turn["ts"],
            last_turn_id=last_turn["id"],
        )
        if self._debug:
            self._logger.debug(
                "Memory summary stored len=%s last_turn_id=%s",
                len(updated),
                last_turn["id"],
            )
        self._maybe_queue_summary()

    def _retrieve_memories(self, query_text: str) -> List[MemoryHit]:
        if not self._store:
            return []

        since_ts = self._since_timestamp()
        candidates = self._store.fetch_turns_since(
            since_ts,
            limit=self._candidate_limit,
            ascending=False,
        )
        if not candidates:
            return []

        if not self._embedder:
            return self._fallback_recent_hits(candidates)

        query_vec = self._embedder.embed(query_text)
        if query_vec is None or query_vec.size == 0:
            return []

        hits: List[MemoryHit] = []
        for row in candidates:
            embedding = row["embedding"]
            if not embedding:
                continue
            vec = np.frombuffer(embedding, dtype=np.float32)
            if vec.size != query_vec.size:
                continue
            score = float(np.dot(query_vec, vec))
            hits.append(
                MemoryHit(
                    ts=row["ts"],
                    role=row["role"],
                    text=row["text"],
                    score=score,
                )
            )

        if not hits:
            return []

        hits.sort(key=lambda item: item.score, reverse=True)
        if self._debug:
            self._logger.debug(
                "Memory retrieval candidates=%s hits=%s top_k=%s",
                len(candidates),
                len(hits),
                self._top_k,
            )
        return hits[: self._top_k]

    def _fallback_recent_hits(self, rows: List[Tuple]) -> List[MemoryHit]:
        hits = []
        for row in rows[: self._top_k]:
            hits.append(
                MemoryHit(
                    ts=row["ts"],
                    role=row["role"],
                    text=row["text"],
                    score=0.0,
                )
            )
        return hits

    def _format_context(self, summary: str, memories: List[MemoryHit], max_chars: int) -> str:
        parts = []
        summary = (summary or "").strip()
        if summary:
            parts.append(f"Long-term summary:\n{summary}")

        if memories:
            lines = []
            for item in memories:
                role = item.role.capitalize()
                text = item.text.replace("\n", " ").strip()
                if not text:
                    continue
                lines.append(f"- {role}: {text}")
            if lines:
                parts.append("Relevant memories:\n" + "\n".join(lines))

        if not parts:
            return ""

        text = "\n\n".join(parts)
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text

        return self._truncate_context(parts, max_chars)

    def _truncate_context(self, parts: List[str], max_chars: int) -> str:
        summary_block = parts[0] if parts else ""
        if len(summary_block) >= max_chars:
            return self._truncate(summary_block, max_chars)

        if len(parts) == 1:
            return self._truncate(summary_block, max_chars)

        remaining = max_chars - len(summary_block) - 2
        if remaining <= 0:
            return self._truncate(summary_block, max_chars)

        memory_block = parts[1]
        memory_block = self._truncate(memory_block, remaining)
        return summary_block + "\n\n" + memory_block

    def _format_turns_for_summary(self, rows: List[Tuple]) -> str:
        lines = []
        for row in rows:
            role = "User" if row["role"] == "user" else "Assistant"
            text = row["text"].replace("\n", " ").strip()
            if not text:
                continue
            if self._summary_turn_chars > 0 and len(text) > self._summary_turn_chars:
                text = text[: max(0, self._summary_turn_chars - 3)] + "..."
            lines.append(f"{role}: {text}")
        return "\n".join(lines)

    def _since_timestamp(self) -> Optional[str]:
        if self._recency_days <= 0:
            return None
        cutoff = datetime.utcnow() - timedelta(days=self._recency_days)
        return cutoff.isoformat(timespec="seconds")

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name, str(default)).strip().lower()
        return raw in {"1", "true", "yes", "on"}
