from faster_whisper import WhisperModel

class Whisper:
    def __init__(self, model_size="medium"):
        self.model_size = model_size
        self.model = None

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(audio_path)
        return " ".join(seg.text for seg in segments)
