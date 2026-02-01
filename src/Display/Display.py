import logging
import math
import threading
import time
import tkinter as tk
import os
from datetime import datetime
from typing import Optional


class Display:
    """Futuristic minimal display for Jarvis with Apple Intelligence-style rainbow border."""

    def __init__(self, title: str = "JARVIS", fullscreen: bool = True):
        self._logger = logging.getLogger(__name__)
        self._ui_thread_id = threading.get_ident()
        self._on_close = None
        self._root = tk.Tk()
        self._root.title(title)
        self._root.configure(bg="#000000")
        self._root.attributes("-fullscreen", fullscreen)
        self._root.bind("<Escape>", lambda _event: self._handle_close())
        self._root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._show_browser_overlay = os.getenv("BROWSER_SHOW_OVERLAY", "true").lower() in {
            "1",
            "true",
            "yes",
        }

        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        if not fullscreen:
            win_w = int(screen_w * 0.8)
            win_h = int(screen_h * 0.8)
            x = max(0, (screen_w - win_w) // 2)
            y = max(0, (screen_h - win_h) // 2)
            self._root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        else:
            self._root.geometry(f"{screen_w}x{screen_h}+0+0")

        self._canvas = tk.Canvas(
            self._root,
            highlightthickness=0,
            bd=0,
            bg="#000000",
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # CRITICAL FIX: Force the UI to update to get the exact rendered dimensions.
        # This prevents the border from being drawn off-screen if the OS window sizing differs slightly.
        self._root.update()
        self._width = self._canvas.winfo_width()
        self._height = self._canvas.winfo_height()

        self._build_rainbow_border()
        self._build_header(title)
        self._build_clock()
        self._build_weather()
        self._build_agenda()
        self._build_status()
        self._build_device_status()
        self._build_caption()
        self._build_boot()
        self._build_browser_overlay()
        self._build_code_overlay()

        self._mode = "idle"
        self._border_phase = 0.0
        self._border_visible = False
        self._last_frame = time.time()

        self._set_border_visible(False)
        self._animate()
        self._update_clock()

    def run(self) -> None:
        self._root.mainloop()

    def close(self) -> None:
        self._handle_close()

    def bind_on_close(self, callback) -> None:
        self._on_close = callback

    def set_mode(self, mode: str) -> None:
        mode = mode.lower().strip()
        if mode not in {"idle", "listening", "speaking", "thinking"}:
            mode = "idle"
        self._logger.debug("Display mode: %s", mode)
        self._run_on_ui(lambda: self._apply_mode(mode))

    def set_caption(self, text: str) -> None:
        self._logger.debug("Display caption: %s", text)
        self._run_on_ui(lambda: self._caption_label.config(text=text))

    def set_weather(
        self, temperature: str, condition: str, location: Optional[str] = None
    ) -> None:
        def apply():
            temp_text = temperature if temperature else "--"
            if not temp_text.endswith("°"):
                temp_text = f"{temp_text}°"
            self._weather_temp.config(text=temp_text)
            self._weather_condition.config(text=condition)
            if location:
                self._weather_location.config(text=location)

        self._logger.debug(
            "Display weather: temp=%s condition=%s location=%s",
            temperature,
            condition,
            location,
        )
        self._run_on_ui(apply)

    def set_agenda(self, items: Optional[list] = None, text: str = "") -> None:
        self._logger.debug("Display agenda items: %s", items)

        def apply():
            for label in self._agenda_lines:
                label.destroy()
            self._agenda_lines.clear()
            if items:
                self._agenda_empty.pack_forget()
                for item in items:
                    line = tk.Label(
                        self._agenda_frame,
                        text=item.get("text", ""),
                        fg=item.get("color", "#FFFFFF"),
                        bg="#000000",
                        font=("SF Pro Display", 16, "normal"),
                        justify="right",
                        wraplength=int(self._width * 0.32),
                    )
                    line.pack(anchor="e")
                    self._agenda_lines.append(line)
            else:
                self._agenda_empty.config(text=text)
                self._agenda_empty.pack(anchor="e")

        self._run_on_ui(apply)

    def set_system_status(self, text: str) -> None:
        self._logger.debug("Display system status: %s", text)
        self._run_on_ui(lambda: self._system_status.config(text=text))

    def set_device_statuses(self, statuses: dict) -> None:
        def apply():
            for key, row in self._device_rows.items():
                value = statuses.get(key)
                color = "#FFD43B"
                if isinstance(value, str):
                    value = value.lower()
                if value is True or value == "on":
                    color = "#4DFF91"
                elif value is False or value == "off":
                    color = "#FF6B6B"
                row["canvas"].itemconfig(row["dot"], fill=color, outline=color)

        self._run_on_ui(apply)

    def set_browser_overlay(
        self, visible: bool, url: str = "", title: str = "Browser"
    ) -> None:
        self._logger.debug("Display browser overlay: %s %s", visible, url)

        def apply():
            self._show_browser_overlay = os.getenv("BROWSER_SHOW_OVERLAY", "true").lower() in {
                "1",
                "true",
                "yes",
            }
            if not self._show_browser_overlay:
                self._browser_frame.place_forget()
                return
            if visible:
                self._update_browser_overlay_geometry()
                self._browser_title.config(text=title)
                self._browser_url.config(text=url)
                self._browser_frame.place(relx=0.5, rely=0.5, anchor="center")
                self._browser_frame.lift()
            else:
                self._browser_frame.place_forget()

        self._run_on_ui(apply)

    def set_code_overlay(self, visible: bool, text: str = "", title: str = "Code") -> None:
        def apply():
            if visible:
                self._code_title.config(text=title)
                self._code_text.configure(state="normal")
                self._code_text.delete("1.0", tk.END)
                self._code_text.insert("1.0", text)
                self._code_text.configure(state="disabled")
                self._code_frame.place(relx=0.5, rely=0.5, anchor="center")
                self._code_frame.lift()
            else:
                self._code_frame.place_forget()

        self._run_on_ui(apply)

    def show_boot(self, lines: list[str], title: str = "Initializing") -> None:
        def apply():
            self._boot_title.config(text=title)
            self._boot_text.config(text="\n".join(lines))
            self._boot_frame.place(relx=0.5, rely=0.72, anchor="center")
            self._boot_frame.lift()

        self._run_on_ui(apply)

    def hide_boot(self) -> None:
        self._run_on_ui(lambda: self._boot_frame.place_forget())

    def _apply_mode(self, mode: str) -> None:
        self._mode = mode
        label = {
            "idle": "Standby",
            "listening": "Listening",
            "speaking": "Speaking",
            "thinking": "Processing",
        }.get(mode, "Standby")
        self._status_label.config(text=label)

        # Show rainbow border for listening, speaking, and thinking modes
        active = mode in {"listening", "speaking", "thinking"}
        self._set_border_visible(active)

    def _run_on_ui(self, fn):
        if threading.get_ident() == self._ui_thread_id:
            fn()
        else:
            self._root.after(0, fn)

    def _build_rainbow_border(self) -> None:
        """Create Apple Intelligence-style rainbow border around the screen."""
        border_width = 8
        # Inset the border by half its width to keep it fully on screen
        inset = border_width // 2

        # Rainbow gradient colors (Apple Intelligence style)
        self._rainbow_colors = [
            "#FF6B9D",  # Pink
            "#FF8E53",  # Orange
            "#FFC644",  # Yellow
            "#4DFF91",  # Green
            "#4DD4FF",  # Cyan
            "#6B8EFF",  # Blue
            "#B47EFF",  # Purple
            "#FF6B9D",  # Back to pink for seamless loop
        ]

        self._border_segments = []

        # 1. Define the Border Box
        x1, y1 = inset, inset
        x2, y2 = self._width - inset, self._height - inset
        
        # Dimensions of the path
        w = x2 - x1
        h = y2 - y1
        perimeter = 2 * (w + h)

        # 2. Generate Drawing Points (FIX FOR SLANTED CORNERS)
        # To avoid slanted corners, we must ensure there is a point EXACTLY at every corner.
        # If we only use fixed intervals, a segment might cut diagonally from 
        # (right_edge, y_near_bottom) to (bottom_edge, x_near_right).

        num_segments = 200
        
        # Calculate regular intervals
        stops = [ (i / num_segments) * perimeter for i in range(num_segments + 1) ]
        
        # Add EXACT corner distances to the list
        # 1. Top-Right corner is at distance 'w'
        stops.append(w)
        # 2. Bottom-Right corner is at distance 'w + h'
        stops.append(w + h)
        # 3. Bottom-Left corner is at distance 'w + h + w'
        stops.append(w + h + w)
        
        # Sort and remove duplicates to create a clean path
        all_distances = sorted(list(set(stops)))

        # 3. Create Segments
        for i in range(len(all_distances) - 1):
            d_start = all_distances[i]
            d_end = all_distances[i+1]
            
            # Get exact X,Y coordinates
            p1 = self._get_border_position(d_start, x1, y1, w, h, perimeter)
            p2 = self._get_border_position(d_end, x1, y1, w, h, perimeter)

            # Create the line segment
            line = self._canvas.create_line(
                p1[0],
                p1[1],
                p2[0],
                p2[1],
                fill=self._rainbow_colors[0],
                width=border_width,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                state="hidden",
            )
            
            self._border_segments.append(
                {
                    "line": line,
                    # We track progress (0.0 to 1.0) for color animation
                    "progress": d_start / perimeter,
                }
            )

        # Store border parameters
        self._border_box = (x1, y1, x2, y2)
        self._border_width = border_width
        self._border_inset = inset

    def _build_header(self, title: str) -> None:
        """Header reserved for future use."""
        self._header_label = None

    def _build_clock(self) -> None:
        """Large centered clock display."""
        self._clock_label = tk.Label(
            self._root,
            text="",
            fg="#FFFFFF",
            bg="#000000",
            font=("SF Pro Display", 120, "bold"),
        )
        self._date_label = tk.Label(
            self._root,
            text="",
            fg="#888888",
            bg="#000000",
            font=("SF Pro Display", 24, "normal"),
        )
        self._clock_label.place(relx=0.5, rely=0.42, anchor="center")
        self._date_label.place(relx=0.5, rely=0.52, anchor="center")

    def _build_weather(self) -> None:
        """Minimal weather display in bottom right."""
        self._weather_frame = tk.Frame(self._root, bg="#000000")
        self._weather_frame.place(relx=0.96, rely=0.94, anchor="se")

        self._weather_temp = tk.Label(
            self._weather_frame,
            text="--°",
            fg="#FFFFFF",
            bg="#000000",
            font=("SF Pro Display", 42, "bold"),
        )
        self._weather_temp.pack(anchor="e")

        self._weather_condition = tk.Label(
            self._weather_frame,
            text="",
            fg="#888888",
            bg="#000000",
            font=("SF Pro Display", 16, "normal"),
        )
        self._weather_condition.pack(anchor="e")

        self._weather_location = tk.Label(
            self._weather_frame,
            text="",
            fg="#666666",
            bg="#000000",
            font=("SF Pro Display", 12, "normal"),
        )
        self._weather_location.pack(anchor="e")

    def _build_agenda(self) -> None:
        """Agenda display in top right."""
        self._agenda_frame = tk.Frame(self._root, bg="#000000")
        self._agenda_frame.place(relx=0.96, rely=0.05, anchor="ne")

        self._agenda_title = tk.Label(
            self._agenda_frame,
            text="Today's Plans",
            fg="#888888",
            bg="#000000",
            font=("SF Pro Display", 12, "normal"),
        )
        self._agenda_title.pack(anchor="e")

        self._agenda_lines = []
        self._agenda_empty = tk.Label(
            self._agenda_frame,
            text="Calendar not configured",
            fg="#FFFFFF",
            bg="#000000",
            font=("SF Pro Display", 16, "normal"),
            justify="right",
            wraplength=int(self._width * 0.32),
        )
        self._agenda_empty.pack(anchor="e")

    def _build_status(self) -> None:
        """Status indicator in top left."""
        self._status_frame = tk.Frame(self._root, bg="#000000")
        self._status_frame.place(relx=0.04, rely=0.05, anchor="nw")

        self._system_status = tk.Label(
            self._status_frame,
            text="System Online",
            fg="#666666",
            bg="#000000",
            font=("SF Pro Display", 11, "normal"),
        )
        self._system_status.pack(anchor="w")

        self._status_label = tk.Label(
            self._status_frame,
            text="Standby",
            fg="#FFFFFF",
            bg="#000000",
            font=("SF Pro Display", 14, "bold"),
        )
        self._status_label.pack(anchor="w")

    def _build_device_status(self) -> None:
        """Subsystem status indicator in bottom left."""
        self._device_frame = tk.Frame(self._root, bg="#000000")
        self._device_frame.place(relx=0.04, rely=0.94, anchor="sw")

        title = tk.Label(
            self._device_frame,
            text="SYSTEMS",
            fg="#888888",
            bg="#000000",
            font=("SF Pro Display", 12, "normal"),
        )
        title.pack(anchor="w")

        self._device_rows = {}
        for key, label_text in [
            ("llm", "LLM"),
            ("stt", "STT"),
            ("tts", "TTS"),
            ("camera", "CAMERA"),
        ]:
            row = tk.Frame(self._device_frame, bg="#000000")
            row.pack(anchor="w", pady=2)

            canvas = tk.Canvas(
                row, width=10, height=10, bg="#000000", highlightthickness=0
            )
            dot = canvas.create_oval(2, 2, 8, 8, fill="#FFD43B", outline="#FFD43B")
            canvas.pack(side="left")

            label = tk.Label(
                row,
                text=label_text,
                fg="#FFFFFF",
                bg="#000000",
                font=("SF Pro Display", 12, "normal"),
            )
            label.pack(side="left", padx=(6, 0))

            self._device_rows[key] = {"canvas": canvas, "dot": dot}

    def _build_caption(self) -> None:
        """Caption text below clock."""
        self._caption_label = tk.Label(
            self._root,
            text="",
            fg="#AAAAAA",
            bg="#000000",
            font=("SF Pro Display", 18, "normal"),
            wraplength=self._width * 0.6,
        )
        self._caption_label.place(relx=0.5, rely=0.62, anchor="center")

    def _build_boot(self) -> None:
        """Boot status overlay."""
        self._boot_frame = tk.Frame(self._root, bg="#000000")

        self._boot_title = tk.Label(
            self._boot_frame,
            text="Initializing",
            fg="#AAAAAA",
            bg="#000000",
            font=("SF Pro Display", 14, "bold"),
        )
        self._boot_title.pack(anchor="center")

        self._boot_text = tk.Label(
            self._boot_frame,
            text="",
            fg="#FFFFFF",
            bg="#000000",
            font=("SF Pro Display", 14, "normal"),
            justify="center",
        )
        self._boot_text.pack(anchor="center")

    def _build_browser_overlay(self) -> None:
        """Placeholder overlay for the browser window - centered with proper aspect ratio."""
        self._browser_scale = 0.7
        geometry = self._compute_browser_geometry()
        browser_width = geometry["width"]
        browser_height = geometry["height"]

        self._browser_frame = tk.Frame(
            self._root,
            bg="#05070A",
            highlightthickness=2,
            highlightbackground="#4DD4FF",
        )
        self._browser_frame.place_forget()
        self._browser_frame.configure(width=browser_width, height=browser_height)
        self._browser_frame.pack_propagate(False)

        self._browser_title = tk.Label(
            self._browser_frame,
            text="Browser",
            fg="#9FC8FF",
            bg="#05070A",
            font=("SF Pro Display", 14, "bold"),
        )
        self._browser_title.pack(anchor="nw", padx=12, pady=(10, 2))

        self._browser_url = tk.Label(
            self._browser_frame,
            text="",
            fg="#888888",
            bg="#05070A",
            font=("SF Pro Display", 12, "normal"),
            wraplength=max(browser_width - 24, 200),
            justify="left",
        )
        self._browser_url.pack(anchor="nw", padx=12)
        self._browser_hint = tk.Label(
            self._browser_frame,
            text="External browser window active",
            fg="#666666",
            bg="#05070A",
            font=("SF Pro Display", 12, "normal"),
        )
        self._browser_hint.pack(anchor="nw", padx=12, pady=(6, 12))

    def get_browser_geometry(self) -> dict:
        """Return absolute screen coords for the browser overlay."""
        if threading.get_ident() == self._ui_thread_id:
            return self._compute_browser_geometry()
        return self._browser_geometry if hasattr(self, "_browser_geometry") else {}

    def _compute_browser_geometry(self) -> dict:
        self._root.update_idletasks()
        width = self._canvas.winfo_width() or self._width
        height = self._canvas.winfo_height() or self._height
        scale_factor = getattr(self, "_browser_scale", 0.7)

        browser_width = int(width * scale_factor)
        browser_height = int(height * scale_factor)

        root_x = self._root.winfo_rootx()
        root_y = self._root.winfo_rooty()
        x = int(root_x + (width - browser_width) / 2)
        y = int(root_y + (height - browser_height) / 2)

        self._browser_geometry = {"x": x, "y": y, "width": browser_width, "height": browser_height}
        return self._browser_geometry

    def _update_browser_overlay_geometry(self) -> None:
        geometry = self._compute_browser_geometry()
        self._browser_frame.configure(width=geometry["width"], height=geometry["height"])
        self._browser_url.config(wraplength=max(geometry["width"] - 24, 200))

    def _build_code_overlay(self) -> None:
        """Overlay for code window."""
        width = int(self._width * 0.7)
        height = int(self._height * 0.6)

        self._code_frame = tk.Frame(
            self._root,
            bg="#05070A",
            highlightthickness=2,
            highlightbackground="#69DB7C",
        )
        self._code_frame.place_forget()
        self._code_frame.configure(width=width, height=height)
        self._code_frame.pack_propagate(False)

        self._code_title = tk.Label(
            self._code_frame,
            text="Code",
            fg="#BFF9C7",
            bg="#05070A",
            font=("SF Pro Display", 14, "bold"),
        )
        self._code_title.pack(anchor="nw", padx=12, pady=(10, 6))

        self._code_text = tk.Text(
            self._code_frame,
            bg="#06090F",
            fg="#E6F2FF",
            insertbackground="#E6F2FF",
            font=("SF Mono", 12, "normal"),
            wrap="none",
            bd=0,
            highlightthickness=0,
        )
        self._code_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._code_text.configure(state="disabled")

    def _set_border_visible(self, visible: bool) -> None:
        """Show or hide the rainbow border."""
        self._border_visible = visible
        state = "normal" if visible else "hidden"
        for segment in self._border_segments:
            self._canvas.itemconfig(segment["line"], state=state)

    def _update_border_colors(self) -> None:
        """Update the gradient flow around the border."""
        if not self._border_visible:
            return

        for segment in self._border_segments:
            # Get color based on position and animation phase
            color_progress = (segment["progress"] + self._border_phase) % 1.0
            color_index = color_progress * (len(self._rainbow_colors) - 1)
            color_idx = int(color_index)
            color_frac = color_index - color_idx

            # Interpolate between colors
            color = self._interpolate_color(
                self._rainbow_colors[color_idx],
                self._rainbow_colors[color_idx + 1],
                color_frac,
            )
            self._canvas.itemconfig(segment["line"], fill=color)

    def _get_border_position(
        self, distance: float, x1: float, y1: float, width: float, height: float, perimeter: float
    ) -> tuple:
        """Calculate x, y position along the border rectangle at given distance."""
        distance = distance % perimeter

        # Top edge (left to right)
        if distance <= width:
            return (x1 + distance, y1)
        distance -= width

        # Right edge (top to bottom)
        if distance <= height:
            return (x1 + width, y1 + distance)
        distance -= height

        # Bottom edge (right to left)
        if distance <= width:
            return (x1 + width - distance, y1 + height)
        distance -= width

        # Left edge (bottom to top)
        return (x1, y1 + height - distance)

    def _interpolate_color(self, color1: str, color2: str, t: float) -> str:
        """Interpolate between two hex colors."""
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)

        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _animate(self) -> None:
        """Main animation loop."""
        now = time.time()
        delta = now - self._last_frame
        self._last_frame = now

        # Different animation speeds for different modes
        phase_speed = 0.0

        if self._mode == "listening":
            phase_speed = 0.35
        elif self._mode == "speaking":
            phase_speed = 0.65
        elif self._mode == "thinking":
            phase_speed = 0.2

        if self._border_visible:
            self._border_phase += delta * phase_speed

            # Keep values in reasonable range
            if self._border_phase > 1.0:
                self._border_phase -= 1.0

            self._update_border_colors()

        self._root.after(33, self._animate)  # ~30 FPS

    def _update_clock(self) -> None:
        """Update clock display every second."""
        now = datetime.now()
        self._clock_label.config(text=now.strftime("%I:%M"))
        self._date_label.config(text=now.strftime("%A, %B %d"))
        self._root.after(1000, self._update_clock)

    def _handle_close(self) -> None:
        """Handle window close event."""
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        self._root.destroy()
