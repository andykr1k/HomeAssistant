from SpeechToText.STT import STT
from TextToSpeech.TTS import TTS
from VisionLanguageModel.VLM import VLM
from Tools.Tools import Tools
from State.State import State
from API.API import API
from Tracking.Tracking import Tracking

class HomeAssistant:
    """
    Central class that coordinates speech, vision, language, tools, and state management.
    """

    def __init__(self):
        # Initialize each subsystem
        print("[HomeAssistant] Initializing subsystems...")
        self.stt = STT(model_size="small", device="cuda:0")
        self.tts = TTS()
        self.vlm = VLM()
        self.tools = Tools()
        self.state = State()
        self.api = API()
        self.tracking = Tracking()
        print("[HomeAssistant] All systems initialized successfully.")

    def run(self):
        """
        Main assistant loop.
        For now, this is a simple text-driven loop that can be extended to voice input.
        """
        print("[HomeAssistant] Assistant is now running. Type 'exit' to quit.\n")

        try:
            response =  "Welcome Home Andy!"
            print(f"[HomeAssistant] Assistant: {response}")
            self.tts.speak(response)

        except KeyboardInterrupt:
            print("\n[HomeAssistant] Interrupted by user.")
        except Exception as e:
            print(f"[HomeAssistant] Error: {e}")
        finally:
            print("[HomeAssistant] Shutting down assistant.")


if __name__ == "__main__":
    assistant = HomeAssistant()
    assistant.run()