import nemo.collections.asr as nemo_asr
import torch


class Parakeet:
    def __init__(self, device="cpu"):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None
        self.device = device

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        result = self.model.transcribe([audio_path])
        return result[0]
        
