from SpeechToText.STT import STT
from TextToSpeech.TTS import TTS
from VisionLanguageModel.VLM import VLM
from Tools.Tools import Tools
from State.State import State

class HomeAssistant():
    def __init__(self):
        self.stt = STT()
        self.tts = TTS()
        self.vlm = VLM()
        self.tools = Tools()
        self.state = State()

    def run(self):
        pass

if __name__ == "__main__":
    assistant = HomeAssistant()
    assistant.run()