import nemo.collections.asr as nemo_asr
import torch


class Canary:
    def __init__(self, device="cpu", **kwargs):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None
        self.device = device

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")

    def transcribe(
        self,
        audio_file,
        language=None,
        timestamps=False,
        confidence=False,
        verbose=False,
    ):
        if self.model is None:
            self.download()

        result = self.model.transcribe([audio_file])

        text = result[0] if isinstance(result, list) else str(result)

        output = {
            "text": text,
            "segments": []
        }

        if timestamps:
            output["segments"].append({
                "start": 0.0,
                "end": 0.0,
                "text": text,
                "confidence": None
            })

        if verbose:
            print("=== CANARY DEBUG ===")
            print(output)

        return output
