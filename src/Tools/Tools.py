from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence
from zoneinfo import ZoneInfo
from urllib.parse import quote_plus
from pathlib import Path

import requests
import webbrowser


@dataclass
class ToolResult:
    ok: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    silent: bool = False


WEATHER_CODES: Dict[int, str] = {
    0: "Clear",
    1: "Mostly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Heavy showers",
    95: "Thunderstorm",
}


class WeatherService:
    """Optional weather fetcher using Open-Meteo."""

    def __init__(self, state, refresh_minutes: int = 10, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._refresh = max(refresh_minutes, 2) * 60
        self._thread = None
        self._running = False
        self._lat = os.getenv("WEATHER_LAT", "")
        self._lon = os.getenv("WEATHER_LON", "")
        self._location = os.getenv("WEATHER_LOCATION", "Home")
        self._units = os.getenv("WEATHER_UNITS", "fahrenheit")

    @property
    def configured(self) -> bool:
        return bool(self._lat and self._lon)

    def start(self) -> None:
        if not self._state:
            return
        if not self._lat or not self._lon:
            if self._debug:
                self._logger.debug(
                    "Weather disabled; missing lat/lon. WEATHER_LAT=%s WEATHER_LON=%s",
                    self._lat,
                    self._lon,
                )
            self._state.update(
                weather_temp="--",
                weather_condition="Weather not configured",
                weather_location=self._location,
            )
            return

        if self._debug:
            self._logger.debug(
                "Weather starting lat=%s lon=%s units=%s location=%s refresh=%ss",
                self._lat,
                self._lon,
                self._units,
                self._location,
                self._refresh,
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self._fetch_once()
            time.sleep(self._refresh)

    def _fetch_once(self) -> None:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self._lat,
            "longitude": self._lon,
            "current_weather": True,
            "temperature_unit": self._units,
            "timezone": "auto",
        }

        try:
            response = requests.get(url, params=params, timeout=8)
            response.raise_for_status()
            data = response.json()
            if self._debug:
                self._logger.debug("Weather response: %s", data)
            current = data.get("current_weather") or {}
            temp = current.get("temperature")
            code = current.get("weathercode")
            condition = WEATHER_CODES.get(code, "Conditions unknown")
            temp_text = f"{round(temp)}" if isinstance(temp, (float, int)) else "--"
            self._state.update(
                weather_temp=temp_text,
                weather_condition=condition,
                weather_location=self._location,
            )
        except requests.RequestException as exc:
            if self._debug:
                self._logger.debug("Weather request failed: %s", exc)
            self._state.update(
                weather_temp="--",
                weather_condition="Weather unavailable",
                weather_location=self._location,
            )


class CalendarService:
    """Optional Google Calendar feed reader (ICS)."""

    def __init__(self, state, refresh_minutes: int = 10, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._refresh = max(refresh_minutes, 2) * 60
        self._thread = None
        self._running = False
        self._url = os.getenv("CALENDAR_ICS_URL", "")
        self._sources = self._parse_sources(os.getenv("CALENDAR_ICS_SOURCES", ""))
        self._timezone = os.getenv("CALENDAR_TIMEZONE", "")
        self._max_items = int(os.getenv("CALENDAR_MAX_ITEMS", "5"))
        self._cache = {}
        self._cache_lock = threading.Lock()

    @property
    def configured(self) -> bool:
        return bool(self._sources or self._url)

    def start(self) -> None:
        if not self._state:
            return
        if not self._sources and not self._url:
            if self._debug:
                self._logger.debug("Calendar disabled; missing CALENDAR_ICS_URL(S).")
            self._state.update(agenda_text="Calendar not configured")
            return

        if self._debug:
            self._logger.debug(
                "Calendar starting sources=%s timezone=%s refresh=%ss",
                self._sources or self._url,
                self._timezone,
                self._refresh,
            )
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self._fetch_once()
            time.sleep(self._refresh)

    def _fetch_once(self) -> None:
        items = self.get_events("today", use_cache=False)
        if not items:
            self._state.update(agenda_items=[], agenda_text="No events today")
            return

        agenda_items = [{"text": item["text"], "color": item["color"]} for item in items]
        self._state.update(agenda_items=agenda_items, agenda_text="")

    def _expand_events(self, cal, window_start, window_end) -> Sequence[Any]:
        try:
            from recurring_ical_events import of as recurring_of

            return list(recurring_of(cal).between(window_start, window_end))
        except Exception as exc:
            if self._debug:
                self._logger.debug("Calendar recurrence disabled: %s", exc)
            return [component for component in cal.walk() if component.name == "VEVENT"]

    def _format_events(self, events, window_start, window_end, tz, label, color) -> List[Dict[str, Any]]:
        sortable = []

        for event in events:
            summary = str(event.get("SUMMARY", "Event"))
            dtstart_raw = event.decoded("DTSTART", None)
            dtend_raw = event.decoded("DTEND", None)
            if not dtstart_raw:
                continue

            start_dt, end_dt, all_day = self._normalize_event(
                dtstart_raw,
                dtend_raw,
                tz,
            )
            if not start_dt or not end_dt:
                continue

            if not self._overlaps_window(start_dt, end_dt, window_start, window_end):
                continue

            if all_day:
                line = f"{label} · All day  {summary}"
            else:
                start_label = self._format_time(start_dt)
                end_label = self._format_time(end_dt)
                if start_label == end_label:
                    line = f"{label} · {start_label}  {summary}"
                else:
                    line = f"{label} · {start_label} - {end_label}  {summary}"

            sortable.append({"start": start_dt, "text": line, "color": color})

        return sortable

    @staticmethod
    def _overlaps_window(start_dt, end_dt, window_start, window_end) -> bool:
        return start_dt < window_end and end_dt > window_start

    @staticmethod
    def _format_time(value: datetime) -> str:
        return value.strftime("%I:%M %p").lstrip("0")

    def _normalize_event(self, dtstart_raw, dtend_raw, tz):
        if isinstance(dtstart_raw, datetime):
            start_dt = self._ensure_tz(dtstart_raw, tz)
            all_day = False
        elif isinstance(dtstart_raw, date):
            start_dt = datetime.combine(dtstart_raw, dt_time.min, tz)
            all_day = True
        else:
            return None, None, False

        if dtend_raw is None:
            end_dt = start_dt + timedelta(hours=1)
        elif isinstance(dtend_raw, datetime):
            end_dt = self._ensure_tz(dtend_raw, tz)
        elif isinstance(dtend_raw, date):
            end_dt = datetime.combine(dtend_raw, dt_time.min, tz)
        else:
            end_dt = start_dt + timedelta(hours=1)

        if end_dt <= start_dt:
            end_dt = start_dt + timedelta(hours=1)

        return start_dt, end_dt, all_day

    def _get_timezone(self):
        if self._timezone:
            try:
                return ZoneInfo(self._timezone)
            except Exception as exc:
                if self._debug:
                    self._logger.debug("Calendar timezone error: %s", exc)
        return datetime.now().astimezone().tzinfo

    @staticmethod
    def _today_window(tz):
        now = datetime.now(tz)
        start = datetime.combine(now.date(), dt_time.min, tz)
        end = start + timedelta(days=1)
        return start, end

    @staticmethod
    def _ensure_tz(value: datetime, tz):
        if value.tzinfo is None:
            return value.replace(tzinfo=tz)
        return value.astimezone(tz)

    @staticmethod
    def _parse_sources(raw: str) -> List[Dict[str, str]]:
        if not raw:
            return []
        palette = ["#4DD4FF", "#FF6B9D", "#FFD43B", "#69DB7C", "#B47EFF"]
        sources = []
        entries = []
        for part in raw.split(";"):
            entries.extend([line for line in part.splitlines() if line.strip()])
        for idx, entry in enumerate(entries):
            bits = [item.strip() for item in entry.split("|") if item.strip()]
            if len(bits) == 3:
                label, color, url = bits
            elif len(bits) == 2:
                label, url = bits
                color = palette[idx % len(palette)]
            elif len(bits) == 1:
                label = "Calendar"
                url = bits[0]
                color = palette[idx % len(palette)]
            else:
                continue
            sources.append({"label": label, "color": color, "url": url})
        return sources

    def _fallback_source(self) -> List[Dict[str, str]]:
        if not self._url:
            return []
        return [{"label": "Calendar", "color": "#4DD4FF", "url": self._url}]

    def get_events(self, range_name: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        sources = self._sources or self._fallback_source()
        if not sources:
            return []

        if use_cache:
            cached = self._read_cache(range_name)
            if cached is not None:
                return cached

        try:
            from icalendar import Calendar
        except Exception as exc:
            if self._debug:
                self._logger.debug("Calendar parse unavailable: %s", exc)
            return []

        tz = self._get_timezone()
        window_start, window_end = self._window_for_range(range_name, tz)
        items = []

        for source in sources:
            url = source["url"]
            label = source["label"]
            color = source["color"]
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.text
            except requests.RequestException as exc:
                if self._debug:
                    self._logger.debug("Calendar request failed (%s): %s", label, exc)
                continue

            cal = Calendar.from_ical(data)
            events = self._expand_events(cal, window_start, window_end)
            items.extend(self._format_events(events, window_start, window_end, tz, label, color))

        if not items:
            return []

        items = sorted(items, key=lambda item: item["start"])
        if len(items) > self._max_items:
            remaining = len(items) - self._max_items
            items = items[: self._max_items] + [
                {"text": f"+{remaining} more", "color": "#888888", "start": window_end}
            ]

        self._write_cache(range_name, items)
        return items

    def _window_for_range(self, range_name: str, tz):
        if range_name == "week":
            start = datetime.combine(datetime.now(tz).date(), dt_time.min, tz)
            end = start + timedelta(days=7)
            return start, end
        return self._today_window(tz)

    def _read_cache(self, range_name: str) -> Optional[List[Dict[str, Any]]]:
        with self._cache_lock:
            entry = self._cache.get(range_name)
            if not entry:
                return None
            timestamp, items = entry
            if time.time() - timestamp > self._refresh:
                return None
            return items

    def _write_cache(self, range_name: str, items: List[Dict[str, Any]]) -> None:
        with self._cache_lock:
            self._cache[range_name] = (time.time(), items)


class BrowserService:
    """System browser controller."""

    def __init__(self, state, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)

    @property
    def available(self) -> bool:
        return True

    def open_url(self, url: str, title: str = "Browser") -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            webbrowser.open(url)
        except Exception as exc:
            return ToolResult(ok=False, message=f"Browser failed: {exc}")
        if self._state:
            # Hide overlay when using the system browser.
            self._state.update(browser_visible=False, browser_url=url, browser_title=title)
        return ToolResult(ok=True, message="Opened.", data={"url": url}, silent=True)

    def close(self) -> ToolResult:
        if self._state:
            self._state.update(browser_visible=False, browser_url="", browser_title="")
        return ToolResult(ok=True, message="Closed.", silent=True)

    def search_web(self, query: str, engine: str = "duckduckgo") -> ToolResult:
        engine = (engine or "duckduckgo").lower()
        if engine == "google":
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            title = "Google"
        elif engine == "bing":
            url = f"https://www.bing.com/search?q={quote_plus(query)}"
            title = "Bing"
        else:
            url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            title = "DuckDuckGo"
        return self.open_url(url, title=title)

    def search_youtube(self, query: str) -> ToolResult:
        url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        return self.open_url(url, title="YouTube")

    def click_youtube_result(self, index: int = 1) -> ToolResult:
        return ToolResult(
            ok=False,
            message="Clicking results requires the embedded browser. Open the page manually.",
        )


class CameraStatusService:
    """Poll camera status endpoint."""

    def __init__(self, state, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._url = os.getenv("CAMERA_STATUS_URL", "").strip()
        self._refresh = int(os.getenv("CAMERA_STATUS_POLL_SECONDS", "5"))
        self._running = False
        self._thread = None

    @property
    def configured(self) -> bool:
        return bool(self._url)

    def start(self) -> None:
        if not self._state or not self._url:
            if self._state:
                self._state.update_subsystem_status("camera", None)
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self._fetch_once()
            time.sleep(max(self._refresh, 2))

    def _fetch_once(self) -> None:
        try:
            response = requests.get(self._url, timeout=4)
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status", "")).lower()
            streaming = bool(data.get("streaming"))
            on = status == "on" and streaming
            self._state.update_subsystem_status("camera", on)
        except Exception as exc:
            if self._debug:
                self._logger.debug("Camera status failed: %s", exc)
            if self._state:
                self._state.update_subsystem_status("camera", False)

class HomeAutomationGateway:
    """Thin wrapper around a Home Assistant REST API (optional)."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self.base_url = os.getenv("HOME_ASSISTANT_URL", "").rstrip("/")
        self.token = os.getenv("HOME_ASSISTANT_TOKEN", "")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def call_service(self, domain: str, service: str, data: Dict[str, Any]) -> ToolResult:
        if not self.configured:
            return ToolResult(
                ok=False,
                message="Home Assistant API not configured.",
                data={"missing": ["HOME_ASSISTANT_URL", "HOME_ASSISTANT_TOKEN"]},
            )

        url = f"{self.base_url}/api/services/{domain}/{service}"
        try:
            response = requests.post(url, headers=self._headers(), json=data, timeout=6)
            if response.status_code >= 400:
                self._logger.debug("Home Assistant error %s: %s", response.status_code, response.text)
                return ToolResult(
                    ok=False,
                    message=f"Home Assistant error: {response.status_code}",
                    data={"response": response.text},
                )
            return ToolResult(ok=True, message="Command sent to Home Assistant.", data=response.json())
        except requests.RequestException as exc:
            return ToolResult(ok=False, message=f"Home Assistant request failed: {exc}")


class Tools:
    """Registry for external control actions."""

    def __init__(self, state=None, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._state = state
        self._ha = HomeAutomationGateway()
        self._lights = self._parse_entities(os.getenv("LIGHT_ENTITY_IDS", ""))
        self._tv = os.getenv("TV_ENTITY_ID", "")
        self._thermostat = os.getenv("THERMOSTAT_ENTITY_ID", "")
        self._camera_url = os.getenv("CAMERA_URL", "").strip()
        self._code_root = os.getenv("CODE_ROOT", "").strip()
        self._projects_root = os.getenv("CODE_PROJECTS_ROOT", "").strip()
        self._code_default = os.getenv("CODE_DEFAULT_PATH", "").strip()
        self._code_max_chars = int(os.getenv("CODE_MAX_CHARS", "12000"))
        self._weather = WeatherService(state, debug=debug) if state else None
        self._calendar = CalendarService(state, debug=debug) if state else None
        self._browser = BrowserService(state, debug=debug) if state else None
        self._camera_status = CameraStatusService(state, debug=debug) if state else None
        self._tool_registry = self._build_tool_registry()

        if self._debug:
            self._logger.debug(
                "Tools config lights=%s tv=%s thermostat=%s ha_url=%s",
                self._lights,
                self._tv,
                self._thermostat,
                self._ha.base_url,
            )

    def start_weather(self) -> None:
        if self._weather:
            self._weather.start()

    def stop_weather(self) -> None:
        if self._weather:
            self._weather.stop()

    def start_calendar(self) -> None:
        if self._calendar:
            self._calendar.start()

    def stop_calendar(self) -> None:
        if self._calendar:
            self._calendar.stop()

    def start_status(self) -> None:
        if self._camera_status:
            self._camera_status.start()

    def stop_status(self) -> None:
        if self._camera_status:
            self._camera_status.stop()

    def weather_configured(self) -> bool:
        return bool(self._weather and self._weather.configured)

    def calendar_configured(self) -> bool:
        return bool(self._calendar and self._calendar.configured)

    def camera_status_configured(self) -> bool:
        return bool(self._camera_status and self._camera_status.configured)

    def browser_available(self) -> bool:
        return bool(self._browser and self._browser.available)

    def tool_specs(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": spec["description"],
                "arguments": spec["schema"],
            }
            for name, spec in self._tool_registry.items()
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        arguments = arguments or {}
        spec = self._tool_registry.get(name)
        if not spec:
            return ToolResult(ok=False, message=f"Unknown tool: {name}")
        try:
            return spec["handler"](**arguments)
        except TypeError as exc:
            return ToolResult(ok=False, message=f"Invalid arguments for {name}: {exc}")
        except Exception as exc:
            return ToolResult(ok=False, message=f"{name} failed: {exc}")

    def lights_on(self) -> ToolResult:
        if not self._lights:
            return ToolResult(
                ok=False,
                message="No light entities configured.",
                data={"missing": ["LIGHT_ENTITY_IDS"]},
            )
        return self._ha.call_service("light", "turn_on", {"entity_id": self._lights})

    def lights_off(self) -> ToolResult:
        if not self._lights:
            return ToolResult(
                ok=False,
                message="No light entities configured.",
                data={"missing": ["LIGHT_ENTITY_IDS"]},
            )
        return self._ha.call_service("light", "turn_off", {"entity_id": self._lights})

    def tv_on(self) -> ToolResult:
        if not self._tv:
            return ToolResult(
                ok=False,
                message="No TV entity configured.",
                data={"missing": ["TV_ENTITY_ID"]},
            )
        return self._ha.call_service("media_player", "turn_on", {"entity_id": self._tv})

    def tv_off(self) -> ToolResult:
        if not self._tv:
            return ToolResult(
                ok=False,
                message="No TV entity configured.",
                data={"missing": ["TV_ENTITY_ID"]},
            )
        return self._ha.call_service("media_player", "turn_off", {"entity_id": self._tv})

    def set_temperature(self, temperature: int) -> ToolResult:
        if not self._thermostat:
            return ToolResult(
                ok=False,
                message="No thermostat entity configured.",
                data={"missing": ["THERMOSTAT_ENTITY_ID"]},
            )
        return self._ha.call_service(
            "climate",
            "set_temperature",
            {"entity_id": self._thermostat, "temperature": temperature},
        )

    def thermostat_status(self) -> ToolResult:
        if not self._thermostat:
            return ToolResult(
                ok=False,
                message="No thermostat entity configured.",
                data={"missing": ["THERMOSTAT_ENTITY_ID"]},
            )
        return ToolResult(
            ok=True,
            message="Thermostat configured.",
            data={"entity_id": self._thermostat},
        )

    def get_weather(self) -> ToolResult:
        if not self._state:
            return ToolResult(ok=False, message="State not configured.")
        snapshot = self._state.snapshot()
        return ToolResult(
            ok=True,
            message="Weather retrieved.",
            data={
                "temperature": snapshot.get("weather_temp", "--"),
                "condition": snapshot.get("weather_condition", "Unknown"),
                "location": snapshot.get("weather_location", "Local"),
            },
        )

    def get_calendar(self, range: str = "today") -> ToolResult:
        if not self._calendar:
            return ToolResult(ok=False, message="Calendar not configured.")
        events = self._calendar.get_events(range)
        if not events:
            return ToolResult(ok=True, message="No events found.", data={"range": range, "events": []})
        return ToolResult(
            ok=True,
            message="Calendar retrieved.",
            data={
                "range": range,
                "events": [item["text"] for item in events],
            },
        )

    def browser_open(self, url: str, title: str = "Browser") -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        return self._browser.open_url(url, title=title)

    def browser_search(self, query: str, engine: str = "duckduckgo") -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        return self._browser.search_web(query, engine=engine)

    def browser_youtube(self, query: str) -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        return self._browser.search_youtube(query)

    def browser_click_youtube(self, index: int = 1) -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        return self._browser.click_youtube_result(index=index)

    def browser_close(self) -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        return self._browser.close()

    def code_open(self, path: Optional[str] = None, title: str = "Code") -> ToolResult:
        target = (path or self._code_default or "").strip()
        root = self._resolve_code_root()
        target_path = None

        if not target:
            listing = self._list_projects(root)
            content = "\n".join(listing) if listing else "No projects found."
            if self._state:
                self._state.update(code_visible=True, code_title="Projects", code_text=content)
            return ToolResult(ok=True, message="Opened.", silent=True)

        target_path = self._resolve_project_path(target, root)
        if not target_path:
            return ToolResult(ok=False, message="Project or path not found.")
        if not self._is_within_root(target_path, root):
            return ToolResult(ok=False, message="Path outside allowed root.")
        if not target_path.exists():
            return ToolResult(ok=False, message="Path not found.")

        if target_path.is_dir():
            entries = sorted([p.name + ("/" if p.is_dir() else "") for p in target_path.iterdir()])
            content = "\n".join(entries)
            title = title or f"Directory: {target_path.name}"
        else:
            try:
                content = target_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                return ToolResult(ok=False, message=f"Failed to read file: {exc}")
            if len(content) > self._code_max_chars:
                content = content[: self._code_max_chars] + "\n... (truncated)"
            title = title or target_path.name

        if self._state:
            self._state.update(
                code_visible=True,
                code_title=title,
                code_text=content,
            )
        return ToolResult(ok=True, message="Opened.", silent=True)

    def code_close(self) -> ToolResult:
        if self._state:
            self._state.update(code_visible=False, code_title="", code_text="")
        return ToolResult(ok=True, message="Closed.", silent=True)

    @staticmethod
    def _is_within_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _resolve_code_root(self) -> Path:
        if self._projects_root:
            return Path(self._projects_root).expanduser().resolve()
        if self._code_root:
            return Path(self._code_root).expanduser().resolve()
        return Path.cwd().resolve()

    def _list_projects(self, root: Path) -> List[str]:
        try:
            return sorted([p.name for p in root.iterdir() if p.is_dir()])
        except Exception:
            return []

    def _resolve_project_path(self, target: str, root: Path) -> Optional[Path]:
        raw = target.strip()
        target_path = Path(raw)
        if target_path.is_absolute():
            return target_path.resolve()
        # Direct relative path under root
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate

        # Fuzzy match against project folders in root
        candidates = [p for p in root.iterdir() if p.is_dir()]
        if not candidates:
            return None
        needle = raw.lower()
        scored = []
        for p in candidates:
            name = p.name.lower()
            if name == needle:
                scored.append((0, len(name), p))
            elif name.startswith(needle):
                scored.append((1, len(name), p))
            elif needle in name:
                scored.append((2, len(name), p))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2].resolve()

    def camera_show(self, url: Optional[str] = None) -> ToolResult:
        if not self._browser:
            return ToolResult(ok=False, message="Browser not available.")
        target = (url or self._camera_url).strip()
        if not target:
            return ToolResult(ok=False, message="Camera URL not configured.")
        result = self._browser.open_url(target, title="Security Camera")
        return ToolResult(
            ok=result.ok,
            message=result.message,
            data=result.data,
            silent=result.silent,
        )

    def camera_hide(self) -> ToolResult:
        return self.browser_close()

    @staticmethod
    def _parse_entities(raw: str) -> List[str]:
        return [item.strip() for item in raw.split(",") if item.strip()]

    def _build_tool_registry(self) -> Dict[str, Dict[str, Any]]:
        return {
            "get_weather": {
                "description": "Get current weather for the home location.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.get_weather,
            },
            "get_calendar": {
                "description": "Get calendar events. range can be 'today' or 'week'.",
                "schema": {
                    "type": "object",
                    "properties": {"range": {"type": "string", "enum": ["today", "week"]}},
                },
                "handler": self.get_calendar,
            },
            "lights_on": {
                "description": "Turn on configured lights.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.lights_on,
            },
            "lights_off": {
                "description": "Turn off configured lights.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.lights_off,
            },
            "tv_on": {
                "description": "Turn on the TV.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.tv_on,
            },
            "tv_off": {
                "description": "Turn off the TV.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.tv_off,
            },
            "set_temperature": {
                "description": "Set thermostat temperature in Fahrenheit.",
                "schema": {
                    "type": "object",
                    "properties": {"temperature": {"type": "integer"}},
                    "required": ["temperature"],
                },
                "handler": self._set_temperature_safe,
            },
            "thermostat_status": {
                "description": "Check thermostat configuration.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.thermostat_status,
            },
            "browser_open": {
                "description": "Open a URL in the overlay browser.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["url"],
                },
                "handler": self.browser_open,
            },
            "browser_search": {
                "description": "Search the web using the overlay browser.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "engine": {"type": "string", "enum": ["duckduckgo", "google", "bing"]},
                    },
                    "required": ["query"],
                },
                "handler": self.browser_search,
            },
            "browser_youtube": {
                "description": "Search YouTube for a query.",
                "schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "handler": self.browser_youtube,
            },
            "browser_click_youtube": {
                "description": "Click a YouTube search result by index (1 is first).",
                "schema": {
                    "type": "object",
                    "properties": {"index": {"type": "integer"}},
                },
                "handler": self.browser_click_youtube,
            },
            "browser_close": {
                "description": "Close the overlay browser.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.browser_close,
            },
            "camera_show": {
                "description": "Show the security camera feed in the overlay browser.",
                "schema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                },
                "handler": self.camera_show,
            },
            "camera_hide": {
                "description": "Hide the security camera feed.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.camera_hide,
            },
            "code_open": {
                "description": "Open a project, file, or directory in the code overlay.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "title": {"type": "string"},
                    },
                },
                "handler": self.code_open,
            },
            "code_close": {
                "description": "Close the code overlay.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.code_close,
            },
        }

    def _set_temperature_safe(self, temperature: Any) -> ToolResult:
        try:
            value = int(temperature)
        except (TypeError, ValueError):
            return ToolResult(ok=False, message="Temperature must be an integer.")
        return self.set_temperature(value)
