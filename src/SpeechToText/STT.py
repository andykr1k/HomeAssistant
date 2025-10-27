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
        device="cuda:0",
        compute_type="int8",
        weights_dir="./SpeechToText/weights",
        samplerate=16000,
    ):
        self.device = device
        self.model_size = model_size
        self.compute_type = compute_type
        self.samplerate = samplerate
        self.weights_dir = weights_dir
        self.buffer = queue.Queue()
        self.running = False

        if "cuda" in device and not torch.cuda.is_available():
            print("[STT] ⚠️ CUDA not available, falling back to CPU.")
            self.device = "cpu"

        print(f"[STT] Loading Whisper '{model_size}' on {self.device} ({compute_type})...")
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("[STT] ✅ Model loaded and ready.")

    def _audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for each recorded audio block."""
        if status:
            print("[STT] Audio stream warning:", status)
        self.buffer.put(indata.copy())

    def live_transcribe(self, window_seconds=5, beam_size=5, vad_filter=True):
        """
        Perform live microphone transcription using Whisper.

        Args:
            window_seconds (int): How many seconds of audio to analyze per update.
            beam_size (int): Beam search size for decoding.
            vad_filter (bool): Whether to filter silence.
        """
        self.running = True
        print("[STT] 🎙️ Live transcription started. Press Ctrl+C to stop.")

        def transcriber():
            audio_data = np.zeros((0, 1), dtype=np.float32)
            last_text = ""

            while self.running:
                if not self.buffer.empty():
                    chunk = self.buffer.get()
                    audio_data = np.concatenate((audio_data, chunk), axis=0)

                    if len(audio_data) > window_seconds * self.samplerate:
                        window = audio_data[-window_seconds * self.samplerate :]
                        segments, _ = self.model.transcribe(
                            window,
                            beam_size=beam_size,
                            vad_filter=vad_filter,
                        )
                        text = " ".join([s.text.strip() for s in segments])
                        text = text.strip()

                        if text and text != last_text:
                            print(f"→ {text}")
                            last_text = text

        threading.Thread(target=transcriber, daemon=True).start()

        with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.samplerate):
            try:
                while self.running:
                    sd.sleep(100)
            except KeyboardInterrupt:
                self.stop()

    def stop(self):
        """Stop live transcription cleanly."""
        self.running = False
        print("\n[STT] 🛑 Live transcription stopped.")
