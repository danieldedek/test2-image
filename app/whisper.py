from faster_whisper import WhisperModel

class Whisper:
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.model = None

    def download(self):
        self.model = WhisperModel(self.model_size, device="auto")

    def transcribe(self, audio_path: str) -> str:
        if self.model is None:
            self.download()

        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments)
