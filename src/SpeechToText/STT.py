import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import threading
import torch

class STT:
    def __init__(
        self,
        model_size="small",
        device="cuda",
        compute_type="float16",
        weights_dir="./src/SpeechToText/weights",
        samplerate=16000,
        debug: bool = False,
    ):
        self._debug = debug
        self._logger = logging.getLogger(__name__)
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.samplerate = samplerate
        self.weights_dir = weights_dir
        self.buffer = queue.Queue()
        self.running = False
        self._paused = False

        if "cuda" in device and not torch.cuda.is_available():
            print("[STT] ⚠️ CUDA not available, falling back to CPU.")
            self.device = "cpu"

        if self._debug:
            self._logger.debug(
                "STT config model=%s device=%s compute_type=%s samplerate=%s",
                model_size,
                self.device,
                compute_type,
                samplerate,
            )
        print(f"[STT] Loading Whisper '{model_size}' on {self.device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("[STT] ✅ Model loaded and ready.")

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each recorded audio block."""
        if status:
            print("[STT] Audio stream warning:", status)
        if self._paused:
            return
        self.buffer.put(indata.copy().reshape(-1))

    def live_transcribe(
        self,
        window_seconds=5,
        beam_size=5,
        vad_filter=True,
        on_text=None,
        on_speech_start=None,
        on_speech_end=None,
        on_no_text=None,
        rms_threshold=0.015,
        rms_end_threshold=None,
        silence_seconds=0.6,
        max_utterance_seconds=12.0,
        speech_start_seconds=0.2,
        pre_roll_seconds=0.2,
    ):
        """
        Perform live microphone transcription using Whisper.

        Args:
            window_seconds (int): Unused (kept for compatibility).
            beam_size (int): Beam search size for decoding.
            vad_filter (bool): Whether to filter silence.
            on_text (callable): Optional callback for each new transcription.
            on_speech_start (callable): Optional callback on speech start.
            on_speech_end (callable): Optional callback on speech end.
            on_no_text (callable): Optional callback when no text is detected.
            rms_threshold (float): RMS threshold for speech detection.
            rms_end_threshold (float): RMS threshold for speech end detection.
            silence_seconds (float): Silence duration to end an utterance.
            max_utterance_seconds (float): Hard cap on utterance length.
            speech_start_seconds (float): Minimum speech duration to trigger start.
            pre_roll_seconds (float): Audio to include before speech start.
        """
        self.running = True
        print("[STT] 🎙️ Live transcription started. Press Ctrl+C to stop.")

        def transcriber():
            utterance = np.zeros(0, dtype=np.float32)
            silence_for = 0.0
            speaking = False
            speech_for = 0.0
            pre_roll = np.zeros(0, dtype=np.float32)
            pre_roll_samples = int(self.samplerate * pre_roll_seconds)
            end_threshold = (
                float(rms_end_threshold)
                if rms_end_threshold is not None
                else max(0.001, rms_threshold * 0.7)
            )

            while self.running:
                try:
                    chunk = self.buffer.get(timeout=0.1)
                except queue.Empty:
                    continue

                if chunk.ndim > 1:
                    chunk = chunk.reshape(-1)

                if chunk.size == 0:
                    continue

                rms = float(np.sqrt(np.mean(chunk ** 2)))
                chunk_duration = chunk.size / self.samplerate

                if pre_roll_samples > 0:
                    pre_roll = np.concatenate((pre_roll, chunk), axis=0)
                    if pre_roll.size > pre_roll_samples:
                        pre_roll = pre_roll[-pre_roll_samples:]

                if not speaking:
                    if rms >= rms_threshold:
                        speech_for += chunk_duration
                        if speech_for >= speech_start_seconds:
                            speaking = True
                            silence_for = 0.0
                            utterance = (
                                np.concatenate((pre_roll, chunk), axis=0)
                                if pre_roll.size
                                else chunk.copy()
                            )
                            if self._debug:
                                self._logger.debug(
                                    "Speech start rms=%.5f pre_roll=%.2fs",
                                    rms,
                                    pre_roll.size / self.samplerate,
                                )
                            if on_speech_start:
                                on_speech_start()
                    else:
                        speech_for = 0.0
                else:
                    utterance = np.concatenate((utterance, chunk), axis=0)
                    if rms >= end_threshold:
                        silence_for = 0.0
                    else:
                        silence_for += chunk_duration
                        if silence_for >= silence_seconds:
                            speaking = False
                            speech_for = 0.0
                            if self._debug:
                                self._logger.debug(
                                    "Speech end silence=%.2fs len=%.2fs",
                                    silence_for,
                                    utterance.size / self.samplerate,
                                )
                            if on_speech_end:
                                on_speech_end()
                            self._transcribe_utterance(
                                utterance,
                                beam_size,
                                vad_filter,
                                on_text,
                                on_no_text,
                            )
                            utterance = np.zeros(0, dtype=np.float32)
                            silence_for = 0.0

                if speaking and max_utterance_seconds:
                    if utterance.size / self.samplerate >= max_utterance_seconds:
                        if self._debug:
                            self._logger.debug("Max utterance reached; forcing decode.")
                        speaking = False
                        speech_for = 0.0
                        if on_speech_end:
                            on_speech_end()
                        self._transcribe_utterance(
                            utterance,
                            beam_size,
                            vad_filter,
                            on_text,
                            on_no_text,
                        )
                        utterance = np.zeros(0, dtype=np.float32)
                        silence_for = 0.0

        threading.Thread(target=transcriber, daemon=True).start()

        with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.samplerate):
            try:
                while self.running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.stop()

    def _transcribe_utterance(self, audio, beam_size, vad_filter, on_text, on_no_text) -> None:
        if audio.size < int(self.samplerate * 0.2):
            if on_no_text:
                on_no_text()
            return
        segments, _ = self.model.transcribe(
            audio,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        if not text:
            if on_no_text:
                on_no_text()
            return
        if self._debug:
            self._logger.debug("Transcript: %s", text)
        if on_text:
            on_text(text)
        else:
            print(f"→ {text}")

    def stop(self):
        """Stop live transcription cleanly."""
        self.running = False
        print("\n[STT] 🛑 Live transcription stopped.")

    def pause(self) -> None:
        """Pause capturing and processing audio."""
        self._paused = True

    def resume(self) -> None:
        """Resume capturing and processing audio."""
        self._paused = False
