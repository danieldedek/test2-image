import os
import sys
import pytest

sys.path.insert(0, '/app')

from canary import Canary
from baseASR import BaseASR

class TestCanaryInit:

    def test_is_subclass_of_base_asr(self):
        assert issubclass(Canary, BaseASR)

    def test_default_parameters(self):
        c = Canary()
        assert c.model_name == "nvidia/canary-180m-flash"
        assert c.device == "cpu"
        assert c.strategy == "beam"
        assert c.beam_size == 5
        assert c.len_pen == 1.0
        assert c.language == "cs" or c.language == "en"
        assert c.use_fp16 is False
        assert c.model is None

    def test_custom_parameters(self):
        c = Canary(
            device="cpu",
            strategy="greedy",
            beam_size=3,
            len_pen=0.8,
            language="de",
            use_fp16=False
        )
        assert c.strategy == "greedy"
        assert c.beam_size == 3
        assert c.len_pen == 0.8
        assert c.language == "de"

    def test_model_is_none_before_download(self):
        c = Canary()
        assert c.model is None

    def test_has_download_method(self):
        assert hasattr(Canary, 'download') and callable(Canary.download)

    def test_has_transcribe_method(self):
        assert hasattr(Canary, 'transcribe') and callable(Canary.transcribe)

    @pytest.mark.parametrize("language", ["en", "de", "es", "fr"])
    def test_supported_languages(self, language):
        c = Canary(language=language)
        assert c.language == language

    @pytest.mark.parametrize("strategy", ["greedy", "beam"])
    def test_strategy_variants(self, strategy):
        c = Canary(strategy=strategy)
        assert c.strategy == strategy

    @pytest.mark.parametrize("beam_size", [1, 3, 5, 10])
    def test_beam_size_variants(self, beam_size):
        c = Canary(beam_size=beam_size)
        assert c.beam_size == beam_size

    @pytest.mark.parametrize("len_pen", [0.5, 1.0, 1.5, 2.0])
    def test_len_pen_variants(self, len_pen):
        c = Canary(len_pen=len_pen)
        assert c.len_pen == len_pen

    def test_fp16_false_by_default(self):
        c = Canary()
        assert c.use_fp16 is False

@pytest.mark.slow
@pytest.mark.canary
class TestCanaryInference:

    @pytest.fixture(scope="class")
    def loaded_canary(self):
        c = Canary(device="cpu", strategy="greedy", language="en")
        c.download()
        return c

    def test_download_loads_model(self, loaded_canary):
        assert loaded_canary.model is not None

    def test_transcribe_short_returns_string(self, loaded_canary, short_wav):
        result = loaded_canary.transcribe(short_wav)
        assert isinstance(result, str)

    def test_transcribe_medium_returns_string(self, loaded_canary, medium_wav):
        result = loaded_canary.transcribe(medium_wav)
        assert isinstance(result, str)

    def test_transcribe_long_returns_string(self, loaded_canary, long_wav):
        result = loaded_canary.transcribe(long_wav)
        assert isinstance(result, str)

    def test_transcribe_silent_does_not_crash(self, loaded_canary, silent_wav):
        result = loaded_canary.transcribe(silent_wav)
        assert isinstance(result, str)

    def test_beam_strategy(self, short_wav):
        c = Canary(device="cpu", strategy="beam", beam_size=3, language="en")
        result = c.transcribe(short_wav)
        assert isinstance(result, str)

    def test_greedy_strategy(self, short_wav):
        c = Canary(device="cpu", strategy="greedy", language="en")
        result = c.transcribe(short_wav)
        assert isinstance(result, str)

    @pytest.mark.parametrize("language", ["en", "de", "es", "fr"])
    def test_all_supported_languages(self, language, short_wav):
        c = Canary(device="cpu", strategy="greedy", language=language)
        result = c.transcribe(short_wav)
        assert isinstance(result, str)

    @pytest.mark.parametrize("len_pen", [0.8, 1.0, 1.5])
    def test_len_pen_combinations(self, len_pen, short_wav):
        c = Canary(device="cpu", strategy="beam", beam_size=3, len_pen=len_pen, language="en")
        result = c.transcribe(short_wav)
        assert isinstance(result, str)


@pytest.mark.slow
@pytest.mark.canary
class TestCanaryRealFiles:

    def test_real_files_transcribe(self, real_wav_files):
        if not real_wav_files:
            pytest.skip("No real .wav files found in /app/uploads/")
        c = Canary(device="cpu", strategy="greedy", language="en")
        for fpath in real_wav_files:
            result = c.transcribe(fpath)
            assert isinstance(result, str), f"Failed on {fpath}"
