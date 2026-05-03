import os
import sys
import pytest

sys.path.insert(0, '/app')

from whisper import Whisper
from baseASR import BaseASR

class TestWhisperInit:

    def test_is_subclass_of_base_asr(self):
        assert issubclass(Whisper, BaseASR)

    def test_default_parameters(self):
        w = Whisper()
        assert w.model_size == "medium"
        assert w.device == "auto"
        assert w.compute_type == "auto"
        assert w.beam_size == 5
        assert w.language is None
        assert w.temperature == 0.0
        assert w.vad_filter is True
        assert w.condition_on_previous_text is True
        assert w.best_of == 5
        assert w.model is None

    def test_custom_parameters(self):
        w = Whisper(
            model_size="tiny",
            device="cpu",
            beam_size=3,
            language="en",
            temperature=0.2,
            vad_filter=False,
            condition_on_previous_text=False,
            best_of=3
        )
        assert w.model_size == "tiny"
        assert w.device == "cpu"
        assert w.beam_size == 3
        assert w.language == "en"
        assert w.temperature == 0.2
        assert w.vad_filter is False
        assert w.condition_on_previous_text is False
        assert w.best_of == 3

    def test_model_is_none_before_download(self):
        w = Whisper(model_size="tiny")
        assert w.model is None

    def test_has_download_method(self):
        assert hasattr(Whisper, 'download') and callable(Whisper.download)

    def test_has_transcribe_method(self):
        assert hasattr(Whisper, 'transcribe') and callable(Whisper.transcribe)

    @pytest.mark.parametrize("model_size", ["tiny", "base", "small", "medium", "large-v3"])
    def test_all_model_sizes_accepted(self, model_size):
        w = Whisper(model_size=model_size)
        assert w.model_size == model_size

    @pytest.mark.parametrize("language", ["en", "cs", "de", "fr", "es"])
    def test_language_parameter(self, language):
        w = Whisper(language=language)
        assert w.language == language

    @pytest.mark.parametrize("beam_size", [1, 3, 5, 10])
    def test_beam_size_variants(self, beam_size):
        w = Whisper(beam_size=beam_size)
        assert w.beam_size == beam_size

    @pytest.mark.parametrize("temperature", [0.0, 0.2, 0.5, 1.0])
    def test_temperature_variants(self, temperature):
        w = Whisper(temperature=temperature)
        assert w.temperature == temperature

    @pytest.mark.parametrize("vad_filter", [True, False])
    def test_vad_filter_variants(self, vad_filter):
        w = Whisper(vad_filter=vad_filter)
        assert w.vad_filter == vad_filter

    @pytest.mark.parametrize("condition", [True, False])
    def test_condition_on_previous_text_variants(self, condition):
        w = Whisper(condition_on_previous_text=condition)
        assert w.condition_on_previous_text == condition

@pytest.mark.slow
@pytest.mark.whisper
class TestWhisperInference:

    @pytest.fixture(scope="class")
    def loaded_whisper(self):
        w = Whisper(model_size="tiny", device="cpu", language="en")
        w.download()
        return w

    def test_download_loads_model(self, loaded_whisper):
        assert loaded_whisper.model is not None

    def test_transcribe_short_returns_string(self, loaded_whisper, short_wav):
        result = loaded_whisper.transcribe(short_wav)
        assert isinstance(result, str)

    def test_transcribe_medium_returns_string(self, loaded_whisper, medium_wav):
        """35s file spans two 30s Whisper chunks."""
        result = loaded_whisper.transcribe(medium_wav)
        assert isinstance(result, str)

    def test_transcribe_long_returns_string(self, loaded_whisper, long_wav):
        """90s file spans three 30s Whisper chunks."""
        result = loaded_whisper.transcribe(long_wav)
        assert isinstance(result, str)

    def test_transcribe_silent_does_not_crash(self, loaded_whisper, silent_wav):
        result = loaded_whisper.transcribe(silent_wav)
        assert isinstance(result, str)

    def test_return_segments_returns_list(self, loaded_whisper, short_wav):
        segments = loaded_whisper.transcribe(short_wav, return_segments=True)
        assert isinstance(segments, list)

    def test_segments_have_expected_fields(self, loaded_whisper, short_wav):
        segments = loaded_whisper.transcribe(short_wav, return_segments=True)
        if segments:
            seg = segments[0]
            assert hasattr(seg, 'start')
            assert hasattr(seg, 'end')
            assert hasattr(seg, 'text')
            assert hasattr(seg, 'no_speech_prob')

    def test_segments_have_word_timestamps(self, loaded_whisper, short_wav):
        segments = loaded_whisper.transcribe(short_wav, return_segments=True)
        if segments:
            seg = segments[0]
            assert hasattr(seg, 'words')

    def test_vad_filter_false_does_not_crash(self, short_wav):
        w = Whisper(model_size="tiny", device="cpu", vad_filter=False)
        result = w.transcribe(short_wav)
        assert isinstance(result, str)

    def test_condition_on_previous_text_false(self, short_wav):
        w = Whisper(model_size="tiny", device="cpu", condition_on_previous_text=False)
        result = w.transcribe(short_wav)
        assert isinstance(result, str)

    def test_beam_size_1_greedy(self, short_wav):
        w = Whisper(model_size="tiny", device="cpu", beam_size=1)
        result = w.transcribe(short_wav)
        assert isinstance(result, str)

    def test_temperature_nonzero(self, short_wav):
        w = Whisper(model_size="tiny", device="cpu", temperature=0.5, best_of=3)
        result = w.transcribe(short_wav)
        assert isinstance(result, str)

    def test_segments_start_end_ordered(self, loaded_whisper, medium_wav):
        segments = loaded_whisper.transcribe(medium_wav, return_segments=True)
        for seg in segments:
            assert seg.start <= seg.end

    def test_segments_chronological(self, loaded_whisper, medium_wav):
        segments = loaded_whisper.transcribe(medium_wav, return_segments=True)
        starts = [seg.start for seg in segments]
        assert starts == sorted(starts)

    def test_no_speech_prob_in_range(self, loaded_whisper, short_wav):
        segments = loaded_whisper.transcribe(short_wav, return_segments=True)
        for seg in segments:
            assert 0.0 <= seg.no_speech_prob <= 1.0


@pytest.mark.slow
@pytest.mark.whisper
class TestWhisperRealFiles:

    def test_real_files_transcribe(self, real_wav_files):
        if not real_wav_files:
            pytest.skip("No real .wav files found in /app/uploads/")
        w = Whisper(model_size="tiny", device="cpu", language="en")
        for fpath in real_wav_files:
            result = w.transcribe(fpath)
            assert isinstance(result, str), f"Failed on {fpath}"

    def test_real_files_with_timestamps(self, real_wav_files):
        if not real_wav_files:
            pytest.skip("No real .wav files found in /app/uploads/")
        w = Whisper(model_size="tiny", device="cpu", language="en")
        for fpath in real_wav_files:
            segments = w.transcribe(fpath, return_segments=True)
            assert isinstance(segments, list), f"Failed on {fpath}"
