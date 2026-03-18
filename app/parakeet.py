import nemo.collections.asr as nemo_asr

class Parakeet:
    def __init__(self):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.download()
        return self.model.transcribe([audio_path])[0]
