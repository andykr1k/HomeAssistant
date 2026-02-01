from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
import logging
import os
import re
import shlex
import shutil
import subprocess
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


class WindowPlacement:
    """Compute external window placement using X11 monitor geometry."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._target_monitor = os.getenv("EXTERNAL_TARGET_MONITOR", "").strip()
        self._scale = self._parse_float(os.getenv("EXTERNAL_WINDOW_SCALE", "0.7"), 0.7)

    def geometry(self, scale_override: Optional[float] = None) -> Dict[str, int]:
        override = self._override_geometry()
        if override:
            return override

        monitors = self._read_monitors()
        if not monitors:
            return {"x": 0, "y": 0, "width": 1280, "height": 720}

        monitor = self._select_monitor(monitors)
        scale = scale_override if scale_override is not None else self._scale
        width = max(240, int(monitor["width"] * scale))
        height = max(180, int(monitor["height"] * scale))
        x = monitor["x"] + max((monitor["width"] - width) // 2, 0)
        y = monitor["y"] + max((monitor["height"] - height) // 2, 0)
        return {"x": x, "y": y, "width": width, "height": height}

    def _override_geometry(self) -> Optional[Dict[str, int]]:
        raw_x = os.getenv("EXTERNAL_WINDOW_X")
        raw_y = os.getenv("EXTERNAL_WINDOW_Y")
        raw_w = os.getenv("EXTERNAL_WINDOW_WIDTH")
        raw_h = os.getenv("EXTERNAL_WINDOW_HEIGHT")
        if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
            return None
        try:
            return {
                "x": int(raw_x),
                "y": int(raw_y),
                "width": int(raw_w),
                "height": int(raw_h),
            }
        except ValueError:
            return None

    def _read_monitors(self) -> List[Dict[str, int]]:
        if not shutil.which("xrandr"):
            return []
        try:
            result = subprocess.run(
                ["xrandr", "--listmonitors"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []
        lines = result.stdout.splitlines()
        monitors = []
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            index_raw = parts[0].rstrip(":")
            flags = parts[1]
            primary = "*" in flags
            name = parts[-1]
            geometry_token = None
            for token in parts:
                if "x" in token and "+" in token and "/" in token:
                    geometry_token = token
                    break
            if not geometry_token:
                continue
            match = re.search(r"(\\d+)\\/\\d+x(\\d+)\\/\\d+\\+(\\d+)\\+(\\d+)", geometry_token)
            if not match:
                continue
            try:
                index = int(index_raw)
                width, height, x, y = [int(value) for value in match.groups()]
            except ValueError:
                continue
            monitors.append(
                {
                    "index": index,
                    "name": name,
                    "primary": primary,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )
        return monitors

    def _select_monitor(self, monitors: List[Dict[str, int]]) -> Dict[str, int]:
        target = self._target_monitor
        if target:
            if target.lower() == "primary":
                for monitor in monitors:
                    if monitor.get("primary"):
                        return monitor
            elif target.isdigit():
                idx = int(target)
                for monitor in monitors:
                    if monitor["index"] == idx:
                        return monitor
                if 1 <= idx <= len(monitors):
                    return monitors[idx - 1]
                for monitor in monitors:
                    if monitor["name"].endswith(target):
                        return monitor
            else:
                for monitor in monitors:
                    if monitor["name"] == target:
                        return monitor
                for monitor in monitors:
                    if target.lower() in monitor["name"].lower():
                        return monitor
        for monitor in monitors:
            if monitor.get("primary"):
                return monitor
        return monitors[0]

    @staticmethod
    def _parse_float(raw: str, fallback: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback


class BrowserService:
    """System browser controller."""

    def __init__(self, state, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._external_enabled = os.getenv("BROWSER_EXTERNAL_ENABLE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._external_bin = os.getenv("BROWSER_EXTERNAL_BIN", "google-chrome").strip()
        self._external_args = os.getenv("BROWSER_EXTERNAL_ARGS", "").strip()
        self._external_scale = self._parse_optional_float(os.getenv("BROWSER_EXTERNAL_SCALE", ""))
        self._external_wmclass = os.getenv("BROWSER_EXTERNAL_WMCLASS", "google-chrome").strip()
        self._external_position = os.getenv("BROWSER_EXTERNAL_POSITION", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self._external_position_timeout = self._parse_optional_float(
            os.getenv("BROWSER_EXTERNAL_POSITION_TIMEOUT", "4")
        ) or 4.0
        self._placement = WindowPlacement(debug=debug)
        self._wmctrl = shutil.which("wmctrl")
        self._window_id: Optional[str] = None

    @property
    def available(self) -> bool:
        return True

    def open_url(self, url: str, title: str = "Browser") -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        if self._external_enabled:
            result = self._open_external(url)
            if not result.ok:
                return result
        else:
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

    def _open_external(self, url: str) -> ToolResult:
        binary = self._external_bin or "google-chrome"
        if not self._binary_available(binary):
            return ToolResult(ok=False, message=f"Browser binary not found: {binary}")
        geometry = self._placement.geometry(scale_override=self._external_scale)
        existing_ids = self._list_window_ids() if self._wmctrl else []
        if self._window_id and self._window_alive(self._window_id):
            if self._wmctrl:
                self._activate_window(self._window_id)
            args = [binary, "--new-tab"]
            if self._external_args:
                args.extend(shlex.split(self._external_args))
            args.append(url)
            try:
                subprocess.Popen(args)
            except Exception as exc:
                return ToolResult(ok=False, message=f"Browser failed: {exc}")
            if self._wmctrl and self._external_position:
                self._position_window_id(self._window_id, geometry)
            return ToolResult(ok=True, message="Opened.", data={"url": url}, silent=True)

        args = [
            binary,
            "--new-window",
            f"--window-position={geometry['x']},{geometry['y']}",
            f"--window-size={geometry['width']},{geometry['height']}",
        ]
        if self._external_args:
            args.extend(shlex.split(self._external_args))
        args.append(url)
        try:
            subprocess.Popen(args)
        except Exception as exc:
            return ToolResult(ok=False, message=f"Browser failed: {exc}")
        if self._wmctrl and self._external_position:
            self._window_id = self._capture_window_id(existing_ids)
            if self._window_id:
                self._position_window_id(self._window_id, geometry)
        return ToolResult(ok=True, message="Opened.", data={"url": url}, silent=True)

    @staticmethod
    def _binary_available(binary: str) -> bool:
        if not binary:
            return False
        if os.path.isabs(binary):
            return os.path.exists(binary)
        return bool(shutil.which(binary))

    def _list_window_ids(self) -> List[str]:
        if not self._wmctrl:
            return []
        try:
            result = subprocess.run(
                [self._wmctrl, "-lx"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []
        ids = []
        for line in result.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 3:
                continue
            window_id = parts[0]
            wmclass = parts[2]
            if self._external_wmclass.lower() in wmclass.lower():
                ids.append(window_id)
        return ids

    def _capture_window_id(self, existing_ids: List[str]) -> Optional[str]:
        if not self._wmctrl:
            return None
        deadline = time.time() + max(self._external_position_timeout, 1.0)
        window_id = None
        while time.time() < deadline:
            current_ids = self._list_window_ids()
            new_ids = [item for item in current_ids if item not in existing_ids]
            if new_ids:
                window_id = new_ids[-1]
                break
            if current_ids:
                window_id = current_ids[-1]
            time.sleep(0.2)
        return window_id

    def _position_window_id(self, window_id: str, geometry: Dict[str, int]) -> None:
        if not self._wmctrl or not window_id:
            return
        try:
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-b",
                    "remove,maximized_vert,maximized_horz",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-e",
                    f"0,{geometry['x']},{geometry['y']},{geometry['width']},{geometry['height']}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _activate_window(self, window_id: str) -> None:
        if not self._wmctrl or not window_id:
            return
        try:
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _window_alive(self, window_id: str) -> bool:
        if not self._wmctrl or not window_id:
            return False
        try:
            result = subprocess.run(
                [self._wmctrl, "-l"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return False
        for line in result.stdout.splitlines():
            if line.split(None, 1)[0] == window_id:
                return True
        return False

    @staticmethod
    def _parse_optional_float(raw: str) -> Optional[float]:
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None


class TerminalService:
    """Tool-driven terminal output runner."""

    def __init__(self, state, debug: bool = False) -> None:
        self._state = state
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._enabled = os.getenv("TERMINAL_ENABLE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._default_cwd = os.getenv("TERMINAL_CWD", "").strip()
        self._prompt = os.getenv("TERMINAL_PROMPT", "$").strip() or "$"
        self._timeout = int(os.getenv("TERMINAL_TIMEOUT_SECONDS", "12"))
        self._max_chars = int(os.getenv("TERMINAL_MAX_CHARS", "12000"))
        self._blocked = self._build_blocklist()

    @property
    def available(self) -> bool:
        return self._enabled

    def open(self, title: str = "Terminal") -> ToolResult:
        if not self.available:
            return self._disabled_result()
        if self._state:
            self._state.update(terminal_visible=True, terminal_title=title)
        return ToolResult(ok=True, message="Opened.", silent=True)

    def close(self) -> ToolResult:
        if not self.available:
            return self._disabled_result()
        if self._state:
            self._state.update(terminal_visible=False, terminal_title="")
        return ToolResult(ok=True, message="Closed.", silent=True)

    def clear(self, title: str = "Terminal") -> ToolResult:
        if not self.available:
            return self._disabled_result()
        if self._state:
            self._state.update(terminal_visible=True, terminal_title=title, terminal_text="")
        return ToolResult(ok=True, message="Cleared.", silent=True)

    def run(self, command: str, cwd: Optional[str] = None, title: str = "Terminal") -> ToolResult:
        if not self.available:
            return self._disabled_result()
        cmd = (command or "").strip()
        if not cmd:
            return ToolResult(ok=False, message="Command is empty.")
        run_cwd = (cwd or self._default_cwd).strip() or os.getcwd()
        safe, reason = self._is_command_safe(cmd)
        if not safe:
            entry = self._format_blocked_entry(cmd, run_cwd, reason or "Command blocked for safety.")
            self._append_entry(entry, title)
            return ToolResult(ok=False, message=reason or "Command blocked for safety.")
        start = time.time()
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=run_cwd,
                capture_output=True,
                text=True,
                timeout=max(self._timeout, 1),
            )
            output = (result.stdout or "") + (result.stderr or "")
            entry = self._format_entry(cmd, run_cwd, output, result.returncode, timed_out=False)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            entry = self._format_entry(cmd, run_cwd, output, None, timed_out=True)
        except Exception as exc:
            entry = self._format_entry(cmd, run_cwd, str(exc), None, timed_out=False, errored=True)

        self._append_entry(entry, title)
        elapsed = time.time() - start
        message = f"Command finished in {elapsed:.1f}s."
        return ToolResult(ok=True, message=message, silent=True)

    def _build_blocklist(self) -> List[Dict[str, str]]:
        return [
            {"pattern": r":\(\)\s*\{:\|:&\};:", "reason": "Fork bomb blocked."},
            {"pattern": r"\brm\s+-rf\s+/\s*$", "reason": "Refusing to remove root."},
            {"pattern": r"\brm\s+-rf\s+/\s", "reason": "Refusing to remove root."},
            {"pattern": r"\brm\s+-rf\s+/\*", "reason": "Refusing to remove root."},
            {"pattern": r"\brm\s+-rf\s+~\b", "reason": "Refusing to remove home."},
            {"pattern": r"\brm\s+-rf\s+--no-preserve-root\b", "reason": "Refusing to remove root."},
            {"pattern": r"\bmkfs(\.|\\s|$)", "reason": "Refusing to format disks."},
            {"pattern": r"\bdd\s+if=", "reason": "Refusing to run disk imaging command."},
            {"pattern": r"\b(shutdown|reboot|poweroff|halt)\b", "reason": "Refusing to power off."},
            {"pattern": r"\binit\s+0\b", "reason": "Refusing to power off."},
            {"pattern": r"\bkill\s+-9\s+1\b", "reason": "Refusing to kill init."},
            {"pattern": r"\bchown\s+-r\s+/\b", "reason": "Refusing to change ownership on root."},
            {"pattern": r"\bchmod\s+-r\s+/\b", "reason": "Refusing to change permissions on root."},
        ]

    def _is_command_safe(self, command: str) -> tuple[bool, Optional[str]]:
        lowered = command.lower()
        for entry in self._blocked:
            if re.search(entry["pattern"], lowered):
                return False, entry["reason"]
        return True, None

    def _append_text(self, existing: str, entry: str) -> str:
        combined = f"{existing}{entry}"
        if len(combined) <= self._max_chars:
            return combined
        trimmed = combined[-self._max_chars :]
        return f"... (truncated)\n{trimmed}"

    def _append_entry(self, entry: str, title: str) -> None:
        if not self._state:
            return
        snapshot = self._state.snapshot()
        existing = snapshot.get("terminal_text", "") or ""
        combined = self._append_text(existing, entry)
        self._state.update(
            terminal_visible=True,
            terminal_title=title,
            terminal_text=combined,
        )

    def _format_blocked_entry(self, command: str, cwd: str, reason: str) -> str:
        prompt = f"{self._prompt} "
        header = f"{prompt}{command}  ({cwd})\n"
        body = reason.rstrip()
        if body:
            body = body + "\n"
        return f"{header}{body}[blocked]\n\n"

    def _format_entry(
        self,
        command: str,
        cwd: str,
        output: str,
        exit_code: Optional[int],
        timed_out: bool,
        errored: bool = False,
    ) -> str:
        prompt = f"{self._prompt} "
        header = f"{prompt}{command}  ({cwd})\n"
        if errored:
            status = "error"
        elif timed_out:
            status = "timeout"
        else:
            status = f"exit {exit_code}" if exit_code is not None else "exit ?"
        body = output.rstrip()
        if body:
            body = body + "\n"
        return f"{header}{body}[{status}]\n\n"

    def _disabled_result(self) -> ToolResult:
        message = "Terminal disabled. Set TERMINAL_ENABLE=true and restart."
        self._append_entry(f"{message}\n\n", "Terminal")
        return ToolResult(ok=False, message=message)


class TerminalExternalService:
    """Open a system terminal window (optionally running a command)."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._enabled = os.getenv("TERMINAL_EXTERNAL_ENABLE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._binary = os.getenv("TERMINAL_EXTERNAL_BIN", "gnome-terminal").strip()
        self._args = os.getenv("TERMINAL_EXTERNAL_ARGS", "").strip()
        self._exec_flag = os.getenv("TERMINAL_EXTERNAL_EXEC_FLAG", "-e").strip() or "-e"
        self._template = os.getenv("TERMINAL_EXTERNAL_COMMAND_TEMPLATE", "").strip()
        self._hold_open = os.getenv("TERMINAL_EXTERNAL_HOLD", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self._scale = self._parse_optional_float(os.getenv("TERMINAL_EXTERNAL_SCALE", ""))
        self._wmclass = os.getenv("TERMINAL_EXTERNAL_WMCLASS", "terminal").strip() or "terminal"
        self._title = os.getenv("TERMINAL_EXTERNAL_TITLE", "Jarvis Terminal").strip()
        self._tmux_enabled = os.getenv("TERMINAL_EXTERNAL_TMUX", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self._tmux_session = os.getenv(
            "TERMINAL_EXTERNAL_TMUX_SESSION", "jarvis-terminal"
        ).strip()
        self._position_timeout = self._parse_float(
            os.getenv("TERMINAL_EXTERNAL_POSITION_TIMEOUT", "4"), 4.0
        )
        self._placement = WindowPlacement(debug=debug)
        self._wmctrl = shutil.which("wmctrl")
        self._tmux = shutil.which("tmux")
        self._window_id: Optional[str] = None
        if not self._tmux:
            self._tmux_enabled = False

    @property
    def available(self) -> bool:
        return self._enabled

    def open(self, command: Optional[str] = None) -> ToolResult:
        if not self.available:
            return ToolResult(ok=False, message="External terminal not enabled.")
        if not self._binary_available(self._binary):
            return ToolResult(ok=False, message=f"Terminal binary not found: {self._binary}")
        if self._tmux_enabled and not self._ensure_tmux_session():
            return ToolResult(ok=False, message="Terminal tmux session failed.")

        if self._window_id and self._window_alive(self._window_id):
            if self._wmctrl:
                self._activate_window(self._window_id)
                self._position_window_id(self._window_id)
            if command:
                if self._tmux_enabled:
                    return self._tmux_send(command)
                return self._spawn_with_command(command)
            return ToolResult(ok=True, message="Opened.", silent=True)

        result = self._spawn_terminal_window()
        if not result.ok:
            return result
        if command:
            if self._tmux_enabled:
                return self._tmux_send(command)
            return self._spawn_with_command(command)
        return result

    def run(self, command: str) -> ToolResult:
        return self.open(command=command)

    def _command_args(self, command: Optional[str]) -> List[str]:
        if not command:
            return []
        command = command.strip()
        if not command:
            return []

        if self._hold_open:
            command = f"{command}; exec bash"

        if self._template:
            tokens = shlex.split(self._template)
            rendered = []
            for token in tokens:
                rendered.append(token.replace("{command}", command))
            return rendered

        flag = self._exec_flag.strip()
        if not flag or flag == "--":
            return ["--", "bash", "-lc", command]
        return [flag, "bash", "-lc", command]

    def _spawn_terminal_window(self) -> ToolResult:
        if not self._binary_available(self._binary):
            return ToolResult(ok=False, message=f"Terminal binary not found: {self._binary}")
        existing_ids = self._list_window_ids() if self._wmctrl else []
        args = [self._binary]
        if self._args:
            args.extend(shlex.split(self._args))
        if self._title:
            args.extend(["--title", self._title])
        if self._tmux_enabled:
            attach_command = f"tmux attach -t {self._tmux_session}"
            args.extend(self._command_args(attach_command))
        try:
            subprocess.Popen(args)
        except Exception as exc:
            return ToolResult(ok=False, message=f"Terminal failed: {exc}")

        if self._wmctrl:
            self._window_id = self._capture_window_id(existing_ids)
            if self._window_id:
                self._position_window_id(self._window_id)
        else:
            return ToolResult(
                ok=True,
                message="Opened. Install wmctrl to position the window.",
                silent=False,
            )
        return ToolResult(ok=True, message="Opened.", silent=True)

    def _spawn_with_command(self, command: str) -> ToolResult:
        args = [self._binary]
        if self._args:
            args.extend(shlex.split(self._args))
        if self._title:
            args.extend(["--title", self._title])
        args.extend(self._command_args(command))
        try:
            subprocess.Popen(args)
        except Exception as exc:
            return ToolResult(ok=False, message=f"Terminal failed: {exc}")
        return ToolResult(ok=True, message="Opened.", silent=True)

    def _ensure_tmux_session(self) -> bool:
        if not self._tmux:
            return False
        try:
            result = subprocess.run(
                [self._tmux, "has-session", "-t", self._tmux_session],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return False
        if result.returncode == 0:
            return True
        try:
            subprocess.run(
                [self._tmux, "new-session", "-d", "-s", self._tmux_session],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return False
        return True

    def _tmux_send(self, command: str) -> ToolResult:
        if not self._tmux or not self._ensure_tmux_session():
            return ToolResult(ok=False, message="tmux not available.")
        cmd = command.strip()
        if not cmd:
            return ToolResult(ok=False, message="Command is empty.")
        try:
            subprocess.run(
                [self._tmux, "send-keys", "-t", self._tmux_session, cmd, "C-m"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception as exc:
            return ToolResult(ok=False, message=f"tmux send failed: {exc}")
        return ToolResult(ok=True, message="Command sent.", silent=True)

    def _list_window_ids(self) -> List[str]:
        if not self._wmctrl:
            return []
        try:
            result = subprocess.run(
                [self._wmctrl, "-lx"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []
        ids = []
        for line in result.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 3:
                continue
            window_id = parts[0]
            wmclass = parts[2]
            if self._wmclass.lower() in wmclass.lower():
                ids.append(window_id)
        return ids

    def _capture_window_id(self, existing_ids: List[str]) -> Optional[str]:
        if not self._wmctrl:
            return None
        deadline = time.time() + max(self._position_timeout, 1.0)
        window_id = None
        while time.time() < deadline:
            current_ids = self._list_window_ids()
            new_ids = [item for item in current_ids if item not in existing_ids]
            if new_ids:
                window_id = new_ids[-1]
                break
            if current_ids:
                window_id = current_ids[-1]
            time.sleep(0.2)
        return window_id

    def _position_window_id(self, window_id: str) -> None:
        if not self._wmctrl or not window_id:
            return
        geometry = self._placement.geometry(scale_override=self._scale)
        try:
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-b",
                    "remove,maximized_vert,maximized_horz",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-e",
                    f"0,{geometry['x']},{geometry['y']},{geometry['width']},{geometry['height']}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _activate_window(self, window_id: str) -> None:
        if not self._wmctrl or not window_id:
            return
        try:
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _window_alive(self, window_id: str) -> bool:
        if not self._wmctrl or not window_id:
            return False
        try:
            result = subprocess.run(
                [self._wmctrl, "-l"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return False
        for line in result.stdout.splitlines():
            if line.split(None, 1)[0] == window_id:
                return True
        return False

    @staticmethod
    def _binary_available(binary: str) -> bool:
        if not binary:
            return False
        if os.path.isabs(binary):
            return os.path.exists(binary)
        return bool(shutil.which(binary))

    @staticmethod
    def _parse_optional_float(raw: str) -> Optional[float]:
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_float(raw: str, fallback: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback


class CodeExternalService:
    """Open VS Code in an external window and optionally position it."""

    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self._enabled = os.getenv("CODE_EXTERNAL_ENABLE", "false").lower() in {
            "1",
            "true",
            "yes",
        }
        self._binary = os.getenv("CODE_EXTERNAL_BIN", "code").strip() or "code"
        self._args = os.getenv("CODE_EXTERNAL_ARGS", "").strip()
        self._scale = self._parse_optional_float(os.getenv("CODE_EXTERNAL_SCALE", ""))
        self._wmclass = os.getenv("CODE_EXTERNAL_WMCLASS", "code").strip() or "code"
        self._position_timeout = self._parse_float(
            os.getenv("CODE_EXTERNAL_POSITION_TIMEOUT", "4"), 4.0
        )
        self._placement = WindowPlacement(debug=debug)
        self._wmctrl = shutil.which("wmctrl")
        self._window_id: Optional[str] = None

    @property
    def available(self) -> bool:
        return self._enabled

    def open(self, path: Optional[str] = None) -> ToolResult:
        if not self.available:
            return ToolResult(ok=False, message="VS Code external not enabled.")
        if not self._binary_available(self._binary):
            return ToolResult(ok=False, message=f"VS Code binary not found: {self._binary}")
        if self._window_id and self._window_alive(self._window_id):
            if path:
                args = [self._binary, "--reuse-window"]
                if self._args:
                    args.extend(shlex.split(self._args))
                args.append(path)
                try:
                    subprocess.Popen(args)
                except Exception:
                    pass
            if self._wmctrl:
                self._activate_window(self._window_id)
                self._position_window_id(self._window_id)
            return ToolResult(ok=True, message="Opened.", silent=True)

        existing_ids = self._list_window_ids() if self._wmctrl else []
        args = [self._binary, "--new-window"]
        if self._args:
            args.extend(shlex.split(self._args))
        if path:
            args.append(path)
        try:
            subprocess.Popen(args)
        except Exception as exc:
            return ToolResult(ok=False, message=f"VS Code failed: {exc}")

        if not self._wmctrl:
            return ToolResult(
                ok=True,
                message="Opened. Install wmctrl to position the window.",
                silent=False,
            )

        self._window_id = self._capture_window_id(existing_ids)
        if not self._window_id:
            return ToolResult(
                ok=True,
                message="Opened. Could not position the window.",
                silent=False,
            )
        self._position_window_id(self._window_id)
        return ToolResult(ok=True, message="Opened.", silent=True)

    def _list_window_ids(self) -> List[str]:
        if not self._wmctrl:
            return []
        try:
            result = subprocess.run(
                [self._wmctrl, "-lx"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return []
        ids = []
        for line in result.stdout.splitlines():
            parts = line.split(None, 4)
            if len(parts) < 3:
                continue
            window_id = parts[0]
            wmclass = parts[2]
            if self._wmclass.lower() in wmclass.lower():
                ids.append(window_id)
        return ids

    def _capture_window_id(self, existing_ids: List[str]) -> Optional[str]:
        if not self._wmctrl:
            return None
        deadline = time.time() + max(self._position_timeout, 1.0)
        window_id = None
        while time.time() < deadline:
            current_ids = self._list_window_ids()
            new_ids = [item for item in current_ids if item not in existing_ids]
            if new_ids:
                window_id = new_ids[-1]
                break
            if current_ids:
                window_id = current_ids[-1]
            time.sleep(0.2)
        return window_id

    def _position_window_id(self, window_id: str) -> None:
        if not self._wmctrl or not window_id:
            return
        geometry = self._placement.geometry(scale_override=self._scale)
        try:
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-b",
                    "remove,maximized_vert,maximized_horz",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [
                    self._wmctrl,
                    "-i",
                    "-r",
                    window_id,
                    "-e",
                    f"0,{geometry['x']},{geometry['y']},{geometry['width']},{geometry['height']}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _activate_window(self, window_id: str) -> None:
        if not self._wmctrl or not window_id:
            return
        try:
            subprocess.run(
                [self._wmctrl, "-i", "-a", window_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return

    def _window_alive(self, window_id: str) -> bool:
        if not self._wmctrl or not window_id:
            return False
        try:
            result = subprocess.run(
                [self._wmctrl, "-l"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return False
        for line in result.stdout.splitlines():
            if line.split(None, 1)[0] == window_id:
                return True
        return False

    @staticmethod
    def _binary_available(binary: str) -> bool:
        if not binary:
            return False
        if os.path.isabs(binary):
            return os.path.exists(binary)
        return bool(shutil.which(binary))

    @staticmethod
    def _parse_optional_float(raw: str) -> Optional[float]:
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_float(raw: str, fallback: float) -> float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback


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
        self._terminal = TerminalService(state, debug=debug) if state else None
        self._terminal_external = TerminalExternalService(debug=debug) if state else None
        self._code_external = CodeExternalService(debug=debug) if state else None
        self._camera_status = CameraStatusService(state, debug=debug) if state else None
        self._tool_registry = self._build_tool_registry()
        self._terminal_external_prefer = os.getenv("TERMINAL_EXTERNAL_PREFER", "true").lower() in {
            "1",
            "true",
            "yes",
        }
        self._code_external_prefer = os.getenv("CODE_EXTERNAL_PREFER", "true").lower() in {
            "1",
            "true",
            "yes",
        }

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

    def terminal_available(self) -> bool:
        return bool(self._terminal and self._terminal.available)

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
        if self._code_external and self._code_external.available and self._code_external_prefer:
            target = (path or self._code_default or self._projects_root or "").strip() or None
            return self._code_external.open(path=target)
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

    def code_open_external(self, path: Optional[str] = None) -> ToolResult:
        if not self._code_external:
            return ToolResult(ok=False, message="VS Code external not available.")
        return self._code_external.open(path=path)

    def terminal_open(self, title: str = "Terminal") -> ToolResult:
        if self._terminal_external and self._terminal_external.available and self._terminal_external_prefer:
            return self._terminal_external.open()
        if not self._terminal:
            return ToolResult(ok=False, message="Terminal not available.")
        return self._terminal.open(title=title)

    def terminal_close(self) -> ToolResult:
        if not self._terminal:
            return ToolResult(ok=False, message="Terminal not available.")
        return self._terminal.close()

    def terminal_clear(self, title: str = "Terminal") -> ToolResult:
        if self._terminal_external and self._terminal_external.available and self._terminal_external_prefer:
            return self._terminal_external.open()
        if not self._terminal:
            return ToolResult(ok=False, message="Terminal not available.")
        return self._terminal.clear(title=title)

    def terminal_run(
        self, command: str, cwd: Optional[str] = None, title: str = "Terminal"
    ) -> ToolResult:
        if self._terminal_external and self._terminal_external.available and self._terminal_external_prefer:
            return self._terminal_external.run(command=command)
        if not self._terminal:
            return ToolResult(ok=False, message="Terminal not available.")
        return self._terminal.run(command=command, cwd=cwd, title=title)

    def terminal_open_external(self, command: Optional[str] = None) -> ToolResult:
        if not self._terminal_external:
            return ToolResult(ok=False, message="External terminal not available.")
        return self._terminal_external.open(command=command)

    def terminal_run_external(self, command: str) -> ToolResult:
        if not self._terminal_external:
            return ToolResult(ok=False, message="External terminal not available.")
        return self._terminal_external.run(command=command)

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
            "code_open_external": {
                "description": "Open a project, file, or directory in a VS Code window on the same monitor.",
                "schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
                "handler": self.code_open_external,
            },
            "terminal_open": {
                "description": "Open the terminal overlay.",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                },
                "handler": self.terminal_open,
            },
            "terminal_close": {
                "description": "Close the terminal overlay.",
                "schema": {"type": "object", "properties": {}},
                "handler": self.terminal_close,
            },
            "terminal_clear": {
                "description": "Clear terminal output.",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                },
                "handler": self.terminal_clear,
            },
            "terminal_run": {
                "description": "Run a shell command and display output in the terminal overlay.",
                "schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["command"],
                },
                "handler": self.terminal_run,
            },
            "terminal_open_external": {
                "description": "Open a system terminal window on the same monitor.",
                "schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
                "handler": self.terminal_open_external,
            },
            "terminal_run_external": {
                "description": "Run a command in a system terminal window on the same monitor.",
                "schema": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                "handler": self.terminal_run_external,
            },
        }

    def _set_temperature_safe(self, temperature: Any) -> ToolResult:
        try:
            value = int(temperature)
        except (TypeError, ValueError):
            return ToolResult(ok=False, message="Temperature must be an integer.")
        return self.set_temperature(value)
