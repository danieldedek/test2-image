import nemo.collections.asr as nemo_asr
import torch


class Parakeet:
    def __init__(self):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        result = self.model.transcribe([audio_path])
        return result[0]
        
