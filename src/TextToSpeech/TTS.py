import os
import piper
from piper import PiperVoice
import io

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class TTS:
    def __init__(self, model_name: str = "us-ryan-medium/en_US-ryan-medium.onnx"):
        """Initialize TTS with a Piper model."""
        volume = 1.0
        length = 1.25
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "weights", model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TTS model not found at: {model_path}")
        
        self.voice = PiperVoice.load(model_path)
        
        self._voiceConfig = piper.SynthesisConfig(
            volume=volume,
            length_scale=length,
            noise_scale=0.0,
            noise_w_scale=0.0,
            normalize_audio=False,
        )
        
        print(f"[TTS] Loaded model: {model_name}")
    
    def synthesize_to_file(self, text: str, output_path: str = "output.wav", **kwargs):
        """Synthesize text to a WAV file."""
        kwargs.pop('syn_config', None)
        
        with open(output_path, "wb") as f:
            self.voice.synthesize(text, f, syn_config=self._voiceConfig, **kwargs)
        
        print(f"[TTS] Saved synthesized speech to {output_path}")
    
    def synthesize_to_memory(self, text: str, **kwargs) -> bytes:
        """Synthesize text to a bytes object."""
        kwargs.pop('syn_config', None)
        
        buffer = io.BytesIO()
        self.voice.synthesize(text, buffer, syn_config=self._voiceConfig, **kwargs)
        
        return buffer.getvalue()
    
    def speak(self, text: str, **kwargs):
        """Synthesize text and play it through the default audio device using streaming."""
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError(
                "sounddevice is not installed. Run `pip install sounddevice` to enable playback."
            )
        
        kwargs.pop('syn_config', None)
        
        stream = sd.OutputStream(
            samplerate=self.voice.config.sample_rate,
            channels=1,
            dtype='int16'
        )
        stream.start()
        
        try:
            print(f"[TTS] Speaking: {text}")
            audio_chunks = self.voice.synthesize(text, syn_config=self._voiceConfig, **kwargs)
            
            for chunk in audio_chunks:
                audio_data = chunk.audio_int16_array
                stream.write(audio_data)
        finally:
            stream.stop()
            stream.close()
        
        print("[TTS] Playback finished.")
    
    def save_and_play(self, text: str, filename: str = "speech.wav", **kwargs):
        """Synthesize text to a file and play it immediately."""
        self.synthesize_to_file(text, filename, **kwargs)
        self.speak(text, **kwargs)
        print(f"[TTS] Played synthesized speech from {filename}")