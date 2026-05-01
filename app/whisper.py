from faster_whisper import WhisperModel


from baseASR import BaseASR

class Whisper(BaseASR):
    def __init__(
        self,
        model_size="medium",
        device="auto",
        compute_type="auto",
        beam_size=5,
        language=None,
        temperature=0.0,
        vad_filter=True,
        condition_on_previous_text=True,
        best_of=5
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        self.beam_size = beam_size
        self.language = language
        self.temperature = temperature
        self.vad_filter = vad_filter
        self.condition_on_previous_text = condition_on_previous_text
        self.best_of = best_of

        self.model = None

    def download(self):
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(self, audio_path: str, return_segments: bool = False):
        if self.model is None:
            self.download()

        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=self.beam_size,
            language=self.language,
            temperature=self.temperature,
            vad_filter=self.vad_filter,
            condition_on_previous_text=self.condition_on_previous_text,
            best_of=self.best_of,
            word_timestamps=True
        )

        segments = list(segments)

        if return_segments:
            return segments

        return " ".join(seg.text for seg in segments)
        
