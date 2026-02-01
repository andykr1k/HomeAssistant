from __future__ import annotations

import logging
import os
import threading
from typing import Optional
from pathlib import Path
from API.API import API
from Display.Display import Display
from LLM.Router import LLMRouter
from SpeechToText.STT import STT
from State.State import State
from TextToSpeech.TTS import TTS
from Tools.Tools import Tools

class HomeAssistant:
    """Central class that coordinates speech, tools, state, and display."""

    def __init__(self, display: Optional[Display] = None) -> None:
        self.display = display
        self._boot_status = {}
        self._boot_order = [
            "Environment",
            "State",
            "Tools",
            "LLM",
            "STT",
            "TTS",
            "Weather",
            "Calendar",
            "Browser",
            "API",
            "Listening",
        ]
        self._boot_update("Environment", "loading")
        self._load_env()
        self._boot_update("Environment", "ready")

        self._debug = self._env_flag("DEBUG")
        self._configure_logging()
        print("[HomeAssistant] Initializing subsystems...")

        self.state = State(debug=self._debug)
        self.state.subscribe(self._on_state_update)
        self._boot_update("State", "ready")

        self.tools = Tools(state=self.state, debug=self._debug)
        self._boot_update("Tools", "ready")
        self._boot_update("Browser", "ready" if self.tools.browser_available() else "disabled")

        self._boot_update("LLM", "loading")
        self.router = self._init_router()
        self._boot_update("LLM", "ready" if self.router else "error")
        self.state.update_subsystem_status("llm", bool(self.router))

        self._busy_lock = threading.Lock()
        self._speaking = False
        self._wake_word = os.getenv("JARVIS_WAKE_WORD", "jarvis").lower()
        self._require_wake_word = os.getenv("JARVIS_REQUIRE_WAKE_WORD", "false").lower() in {
            "1",
            "true",
            "yes",
        }

        self._boot_update("STT", "loading")
        self.stt = self._init_stt()
        self._boot_update("STT", "ready" if self.stt else "error")
        self.state.update_subsystem_status("stt", bool(self.stt))

        self._boot_update("TTS", "loading")
        self.tts = self._init_tts()
        self._boot_update("TTS", "ready" if self.tts else "error")
        self.state.update_subsystem_status("tts", bool(self.tts))

        self._boot_update("API", "loading")
        self.api = API(state=self.state, command_handler=self.handle_text, debug=self._debug)

        self._api_thread = None
        self._stt_thread = None
        print("[HomeAssistant] All systems initialized successfully.")

    def start(self) -> None:
        if self._require_wake_word:
            self.state.update(mode="idle", caption=f"Say '{self._wake_word}' to wake")
        else:
            self.state.update(mode="idle", caption="Awaiting command")

        if self.tools.weather_configured():
            self._boot_update("Weather", "starting")
            self.tools.start_weather()
            self._boot_update("Weather", "running")
        else:
            self._boot_update("Weather", "disabled")

        if self.tools.calendar_configured():
            self._boot_update("Calendar", "starting")
            self.tools.start_calendar()
            self._boot_update("Calendar", "running")
        else:
            self._boot_update("Calendar", "disabled")

        self.tools.start_status()

        self._boot_update("API", "starting")
        self._start_api()
        self._boot_update("API", "running")
        self._start_listening()
        self._boot_update("Listening", "ready" if self.stt else "disabled")
        self._boot_complete()
        if self.display:
            self.display.set_system_status("SYSTEM ONLINE")

    def stop(self) -> None:
        if self.stt:
            self.stt.stop()
        self.tools.stop_weather()
        self.tools.stop_calendar()
        self.tools.stop_status()
        self.state.update(mode="idle", caption="Shutting down")

    def handle_text(self, text: str, source: str = "voice") -> None:
        if not text:
            return
        thread = threading.Thread(
            target=self._process_text,
            args=(text, source),
            daemon=True,
        )
        thread.start()

    def _process_text(self, text: str, source: str) -> None:
        if not self._busy_lock.acquire(blocking=False):
            return
        try:
            clean_text = text.strip()
            if self._require_wake_word and self._wake_word not in clean_text.lower():
                if self._require_wake_word:
                    self.state.update(mode="idle", caption=f"Say '{self._wake_word}' to wake")
                else:
                    self.state.update(mode="idle", caption="Awaiting command")
                return

            display_text = clean_text
            clean_text = clean_text.lower()
            if self._require_wake_word:
                clean_text = clean_text.replace(self._wake_word, "").strip()
                display_text = display_text.replace(self._wake_word, "").strip()

            self.state.update(mode="thinking", last_user_text=clean_text, caption=f"Heard: {display_text}")
            response = self._generate_response(clean_text)
            if response:
                self.state.update(mode="speaking", last_assistant_text=response, caption=response)
                self._speak(response)
            if self._require_wake_word:
                self.state.update(mode="idle", caption=f"Say '{self._wake_word}' to wake")
            else:
                self.state.update(mode="idle", caption="Awaiting command")
        finally:
            self._busy_lock.release()

    def _on_transcript(self, text: str) -> None:
        if self._speaking:
            return
        clean_text = text.strip()
        if not clean_text:
            return
        self.handle_text(clean_text, source="voice")

    def _on_speech_start(self) -> None:
        if self._speaking:
            return
        self.state.update(mode="listening", caption="Listening...")

    def _on_speech_end(self) -> None:
        if self._speaking:
            return
        self.state.update(mode="thinking", caption="Processing...")

    def _on_no_text(self) -> None:
        if self._speaking:
            return
        if self._require_wake_word:
            self.state.update(mode="idle", caption=f"Say '{self._wake_word}' to wake")
        else:
            self.state.update(mode="idle", caption="Awaiting command")

    def _generate_response(self, text: str) -> str:
        if self.router and self.router.enabled():
            return self.router.handle(text)

        return "LLM not configured."

    def _speak(self, text: str) -> None:
        if not self.tts:
            return
        try:
            self._speaking = True
            if self.stt:
                self.stt.pause()
            self.tts.speak(text)
        except Exception as exc:
            self.state.update(mode="idle", caption=f"TTS error: {exc}")
        finally:
            self._speaking = False
            if self.stt:
                self.stt.resume()

    def _init_stt(self) -> Optional[STT]:
        try:
            model_size = os.getenv("STT_MODEL_SIZE", "small")
            device = os.getenv("STT_DEVICE", "cuda:0")
            compute_type = os.getenv("STT_COMPUTE_TYPE", "float16")
            return STT(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                debug=self._debug,
            )
        except Exception as exc:
            print(f"[HomeAssistant] STT unavailable: {exc}")
            self.state.update(caption="STT unavailable")
            return None

    def _init_tts(self) -> Optional[TTS]:
        try:
            model_name = os.getenv("TTS_MODEL", "us-ryan-medium/en_US-ryan-medium.onnx")
            return TTS(model_name=model_name, debug=self._debug)
        except Exception as exc:
            print(f"[HomeAssistant] TTS unavailable: {exc}")
            self.state.update(caption="TTS unavailable")
            return None

    def _init_router(self) -> Optional[LLMRouter]:
        try:
            router = LLMRouter(self.tools, debug=self._debug)
            if router.enabled():
                return router
            return None
        except Exception as exc:
            print(f"[HomeAssistant] LLM unavailable: {exc}")
            return None

    def _load_env(self) -> None:
        try:
            from dotenv import load_dotenv
        except Exception:
            print("[HomeAssistant] python-dotenv not installed; .env will be ignored.")
            return

        root_env = Path(__file__).resolve().parents[1] / ".env"
        src_env = Path(__file__).resolve().parent / ".env"
        if root_env.exists():
            load_dotenv(root_env)
        if src_env.exists():
            load_dotenv(src_env)

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        raw = os.getenv(name, str(default)).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _configure_logging(self) -> None:
        level = logging.DEBUG if self._debug else logging.INFO
        logging.basicConfig(
            level=level,
            format="[%(levelname)s] %(name)s: %(message)s",
        )
        logging.getLogger("uvicorn").setLevel(level)
        logging.getLogger("uvicorn.error").setLevel(level)
        logging.getLogger("uvicorn.access").setLevel(level)

        external_verbose = self._env_flag("DEBUG_EXTERNAL")
        if not external_verbose:
            logging.getLogger("faster_whisper").setLevel(logging.WARNING)
            logging.getLogger("ctranslate2").setLevel(logging.WARNING)

    def _start_api(self) -> None:
        host = os.getenv("API_HOST", "0.0.0.0")
        port = int(os.getenv("API_PORT", "8000"))
        self._api_thread = threading.Thread(
            target=self.api.run,
            kwargs={"host": host, "port": port},
            daemon=True,
        )
        self._api_thread.start()

    def _start_listening(self) -> None:
        if not self.stt:
            self.state.update(mode="idle", caption="STT not configured")
            return
        if self._require_wake_word:
            self.state.update(mode="idle", caption=f"Say '{self._wake_word}' to wake")
        else:
            self.state.update(mode="idle", caption="Awaiting command")
        window_seconds = int(os.getenv("STT_WINDOW_SECONDS", "5"))
        beam_size = int(os.getenv("STT_BEAM_SIZE", "5"))
        vad_filter = os.getenv("STT_VAD_FILTER", "true").lower() in {"1", "true", "yes"}
        rms_threshold = float(os.getenv("STT_RMS_THRESHOLD", "0.015"))
        rms_end_threshold = os.getenv("STT_RMS_END_THRESHOLD")
        silence_seconds = float(os.getenv("STT_SILENCE_SECONDS", "0.6"))
        max_utterance_seconds = float(os.getenv("STT_MAX_UTTERANCE_SECONDS", "12"))
        speech_start_seconds = float(os.getenv("STT_SPEECH_START_SECONDS", "0.2"))
        pre_roll_seconds = float(os.getenv("STT_PRE_ROLL_SECONDS", "0.2"))
        self._stt_thread = threading.Thread(
            target=self.stt.live_transcribe,
            kwargs={
                "on_text": self._on_transcript,
                "on_speech_start": self._on_speech_start,
                "on_speech_end": self._on_speech_end,
                "on_no_text": self._on_no_text,
                "window_seconds": window_seconds,
                "beam_size": beam_size,
                "vad_filter": vad_filter,
                "rms_threshold": rms_threshold,
                "rms_end_threshold": float(rms_end_threshold) if rms_end_threshold else None,
                "silence_seconds": silence_seconds,
                "max_utterance_seconds": max_utterance_seconds,
                "speech_start_seconds": speech_start_seconds,
                "pre_roll_seconds": pre_roll_seconds,
            },
            daemon=True,
        )
        self._stt_thread.start()

    def _boot_update(self, name: str, status: str) -> None:
        self._boot_status[name] = status
        if not self.display:
            return
        lines = [f"{item}: {self._boot_status.get(item, 'pending')}" for item in self._boot_order]
        self.display.show_boot(lines)

    def _boot_complete(self) -> None:
        if self.display:
            self.display.hide_boot()

    def _on_state_update(self, snapshot) -> None:
        if not self.display:
            return
        self.display.set_mode(snapshot.get("mode", "idle"))
        self.display.set_caption(snapshot.get("caption", ""))
        self.display.set_weather(
            snapshot.get("weather_temp", "--"),
            snapshot.get("weather_condition", "Unknown"),
            snapshot.get("weather_location", "Local"),
        )
        agenda_items = snapshot.get("agenda_items") or []
        agenda_text = snapshot.get("agenda_text", "")
        self.display.set_agenda(agenda_items, agenda_text)
        self.display.set_browser_overlay(
            bool(snapshot.get("browser_visible")),
            snapshot.get("browser_url", ""),
            snapshot.get("browser_title", "Browser"),
        )
        self.display.set_code_overlay(
            bool(snapshot.get("code_visible")),
            snapshot.get("code_text", ""),
            snapshot.get("code_title", "Code"),
        )
        self.display.set_device_statuses(snapshot.get("subsystem_status", {}))


if __name__ == "__main__":
    display = Display(fullscreen=True)
    assistant_ref = {"assistant": None}

    def on_close():
        assistant = assistant_ref.get("assistant")
        if assistant:
            assistant.stop()

    display.bind_on_close(on_close)

    def boot():
        assistant = HomeAssistant(display=display)
        assistant_ref["assistant"] = assistant
        assistant.start()

    threading.Thread(target=boot, daemon=True).start()
    display.run()
