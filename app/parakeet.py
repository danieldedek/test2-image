import nemo.collections.asr as nemo_asr
import torch


class Parakeet:
    def __init__(
        self,
        device="cpu",
        strategy="beam",
        beam_size=5,
        alpha=0.5,
        beta=1.0,
        use_fp16=False
    ):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None

        self.device = device
        self.strategy = strategy
        self.beam_size = beam_size
        self.alpha = alpha
        self.beta = beta
        self.use_fp16 = use_fp16

    def download(self):
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        if device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        self.model.change_decoding_strategy({
            "strategy": self.strategy,
            "beam_size": self.beam_size,
            "alpha": self.alpha,
            "beta": self.beta
        })

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        return self.model.transcribe([audio_path])[0]
        
