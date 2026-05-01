import nemo.collections.asr as nemo_asr
import torch
from omegaconf import OmegaConf

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
        self.model = nemo_asr.models.EncDecMultiTaskModel.from_pretrained(self.model_name)
        
        device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)
        
        if device == "cuda" and self.use_fp16:
            self.model = self.model.half()

        decoding_cfg = self.model.cfg.decoding
        with OmegaConf.open_dict(decoding_cfg):
            decoding_cfg.strategy = self.strategy
            if self.strategy == "beam":
                decoding_cfg.beam.beam_size = self.beam_size
                decoding_cfg.beam.len_pen = self.len_pen

        self.model.change_decoding_strategy(decoding_cfg)

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        transcribe_cfg = [{
            "audio_filepath": audio_path,
            "duration": None,
            "taskname": "asr",
            "source_lang": self.language,
            "target_lang": self.language,
            "pnc": "yes",
            "answer": "predict",
        }]

        result = self.model.transcribe(
            transcribe_cfg,
            return_hypotheses=self.return_hypotheses
        )
        return result[0]
        
