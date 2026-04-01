import nemo.collections.asr as nemo_asr
import torch


class Canary:
    def __init__(self, device="cpu"):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None
        self.device = device

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.download()
        return self.model.transcribe([audio_path])[0]
