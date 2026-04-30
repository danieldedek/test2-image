import nemo.collections.asr as nemo_asr


class Parakeet:
    def __init__(self, device="cpu"):
        self.model_name = "nvidia/parakeet-tdt-0.6b-v3"
        self.model = None

    def download(self):
        print("Loading Parakeet model...")

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        self.model.change_decoding_strategy({
            "strategy": "greedy"
        })

    def transcribe(self, audio_path: str):
        if self.model is None:
            self.download()

        result = self.model.transcribe([audio_path])
        return result[0]
        
