import nemo.collections.asr as nemo_asr

class Parakeet:
    def __init__(self):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None

    def transcribe(self, audio_path: str) -> str:
        return self.model.transcribe([audio_path])[0]
