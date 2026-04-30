import nemo.collections.asr as nemo_asr
import torch


class Canary:
    def __init__(
        self,
        device="cpu",
        strategy="beam",
        beam_size=5,
        len_pen=1.0,
        language="cs",
        use_fp16=False,
        return_hypotheses=False
    ):
        self.model_name = "nvidia/canary-180m-flash"
        self.model = None

        self.device = device
        self.strategy = strategy
        self.beam_size = beam_size
        self.len_pen = len_pen
        self.language = language
        self.use_fp16 = use_fp16
        self.return_hypotheses = return_hypotheses

    def download(self):
        print("Loading Canary model...")

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        if device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        self.model.change_decoding_strategy({
            "strategy": self.strategy,
            "beam_size": self.beam_size,
            "len_pen": self.len_pen,
        })

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        result = self.model.transcribe(
            [audio_path],
            language=self.language,
            return_hypotheses=self.return_hypotheses
        )

        return result[0]
        
