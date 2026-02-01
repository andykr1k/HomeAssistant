from __future__ import annotations

from datetime import datetime
import logging
import threading
from typing import Any, Callable, Dict, List


class State:
    """Thread-safe shared state with subscription hooks."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._subscribers: List[Callable[[Dict[str, str]], None]] = []
        self._data: Dict[str, Any] = {
            "mode": "idle",
            "caption": "Awaiting command",
            "last_user_text": "",
            "last_assistant_text": "",
            "weather_temp": "--",
            "weather_condition": "Not configured",
            "weather_location": "Local",
            "agenda_text": "Calendar not configured",
            "agenda_items": [],
            "browser_visible": False,
            "browser_url": "",
            "browser_title": "",
            "code_visible": False,
            "code_title": "",
            "code_text": "",
            "subsystem_status": {
                "llm": None,
                "stt": None,
                "tts": None,
                "camera": None,
                "memory": None,
            },
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers.append(callback)

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            self._data.update(kwargs)
            self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            snapshot = dict(self._data)

        if self._debug:
            self._logger.debug("State update: %s", snapshot)

        for callback in self._subscribers:
            try:
                callback(snapshot)
            except Exception:
                continue

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._data)

    def update_subsystem_status(self, name: str, value: Any) -> None:
        with self._lock:
            current = dict(self._data.get("subsystem_status", {}))
            current[name] = value
            self._data["subsystem_status"] = current
            self._data["updated_at"] = datetime.now().isoformat(timespec="seconds")
            snapshot = dict(self._data)

        if self._debug:
            self._logger.debug("Subsystem status update: %s=%s", name, value)

        for callback in self._subscribers:
            try:
                callback(snapshot)
            except Exception:
                continue
