import nemo.collections.asr as nemo_asr

class Canary:
    def __init__(self):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.download()
        return self.model.transcribe([audio_path])[0]
