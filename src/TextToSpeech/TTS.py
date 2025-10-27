import os
from piper import PiperVoice
import io
import wave
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class TTS:
    def __init__(self, model_name: str = "piper-voices\en\en_US\ryan\medium\en_US-ryan-medium.onnx"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "weights", model_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TTS model not found at: {model_path}")

        self.voice = PiperVoice.load(model_path)
        print(f"[TTS] Loaded model: {model_name}")

    def synthesize_to_file(self, text: str, output_path: str = "output.wav", **kwargs):
        with open(output_path, "wb") as f:
            self.voice.synthesize(text, f, **kwargs)
        print(f"[TTS] Saved synthesized speech to {output_path}")

    def synthesize_to_memory(self, text: str, **kwargs) -> bytes:
        buffer = io.BytesIO()
        self.voice.synthesize(text, buffer, **kwargs)
        return buffer.getvalue()

    def speak(self, text: str, **kwargs):
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice is not installed. Run `pip install sounddevice` to enable playback.")

        buffer = io.BytesIO()
        self.voice.synthesize(text, buffer, **kwargs)
        buffer.seek(0)

        with wave.open(buffer, 'rb') as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            sd.play(audio_data, wf.getframerate())
            sd.wait()
        print("[TTS] Playback finished.")

    def save_and_play(self, text: str, filename: str = "speech.wav", **kwargs):
        self.synthesize_to_file(text, filename, **kwargs)
        self.speak(text, **kwargs)
