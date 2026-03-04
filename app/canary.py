import nemo.collections.asr as nemo_asr

class Canary:
    def __init__(self):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None

    def transcribe(self, audio_path: str) -> str:
        return self.model.transcribe([audio_path])[0]
