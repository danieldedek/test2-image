import os
import sys
import pytest

sys.path.insert(0, '/app')

from parakeet import Parakeet
from baseASR import BaseASR

class TestParakeetInit:

    def test_is_subclass_of_base_asr(self):
        assert issubclass(Parakeet, BaseASR)

    def test_default_parameters(self):
        p = Parakeet()
        assert p.model_name == "nvidia/parakeet-tdt-0.6b-v3"
        assert p.device == "cpu"
        assert p.strategy == "beam"
        assert p.beam_size == 5
        assert p.use_fp16 is False
        assert p.model is None

    def test_custom_parameters(self):
        p = Parakeet(
            device="cpu",
            strategy="greedy",
            beam_size=3,
            use_fp16=False
        )
        assert p.strategy == "greedy"
        assert p.beam_size == 3
        assert p.use_fp16 is False

    def test_model_is_none_before_download(self):
        p = Parakeet()
        assert p.model is None

    def test_has_download_method(self):
        assert hasattr(Parakeet, 'download') and callable(Parakeet.download)

    def test_has_transcribe_method(self):
        assert hasattr(Parakeet, 'transcribe') and callable(Parakeet.transcribe)

    @pytest.mark.parametrize("strategy", ["greedy", "beam"])
    def test_strategy_variants(self, strategy):
        p = Parakeet(strategy=strategy)
        assert p.strategy == strategy

    @pytest.mark.parametrize("beam_size", [1, 3, 5, 10])
    def test_beam_size_variants(self, beam_size):
        p = Parakeet(beam_size=beam_size)
        assert p.beam_size == beam_size

    def test_fp16_false_by_default(self):
        p = Parakeet()
        assert p.use_fp16 is False

@pytest.mark.slow
@pytest.mark.parakeet
class TestParakeetInference:

    @pytest.fixture(scope="class")
    def loaded_parakeet(self):
        p = Parakeet(device="cpu", strategy="greedy")
        p.download()
        return p

    def test_download_loads_model(self, loaded_parakeet):
        assert loaded_parakeet.model is not None

    def test_transcribe_short_returns_string(self, loaded_parakeet, short_wav):
        result = loaded_parakeet.transcribe(short_wav)
        assert isinstance(result, str)

    def test_transcribe_medium_returns_string(self, loaded_parakeet, medium_wav):
        result = loaded_parakeet.transcribe(medium_wav)
        assert isinstance(result, str)

    def test_transcribe_long_returns_string(self, loaded_parakeet, long_wav):
        result = loaded_parakeet.transcribe(long_wav)
        assert isinstance(result, str)

    def test_transcribe_silent_does_not_crash(self, loaded_parakeet, silent_wav):
        result = loaded_parakeet.transcribe(silent_wav)
        assert isinstance(result, str)

    def test_greedy_strategy(self, short_wav):
        p = Parakeet(device="cpu", strategy="greedy")
        result = p.transcribe(short_wav)
        assert isinstance(result, str)

    def test_beam_strategy(self, short_wav):
        p = Parakeet(device="cpu", strategy="beam", beam_size=3)
        result = p.transcribe(short_wav)
        assert isinstance(result, str)

    @pytest.mark.parametrize("beam_size", [1, 3, 5])
    def test_beam_size_combinations(self, beam_size, short_wav):
        p = Parakeet(device="cpu", strategy="beam", beam_size=beam_size)
        result = p.transcribe(short_wav)
        assert isinstance(result, str)

    def test_result_is_not_none(self, loaded_parakeet, short_wav):
        result = loaded_parakeet.transcribe(short_wav)
        assert result is not None


@pytest.mark.slow
@pytest.mark.parakeet
class TestParakeetRealFiles:

    def test_real_files_transcribe(self, real_wav_files):
        if not real_wav_files:
            pytest.skip("No real .wav files found in /app/uploads/")
        p = Parakeet(device="cpu", strategy="greedy")
        for fpath in real_wav_files:
            result = p.transcribe(fpath)
            assert isinstance(result, str), f"Failed on {fpath}"
