import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, '/app')

import utils
from whisper import Whisper
from canary import Canary
from parakeet import Parakeet


@pytest.fixture(autouse=True)
def clear_cache():
    utils._model_cache.clear()
    yield
    utils._model_cache.clear()


class TestCreateAsrEngine:

    def test_unknown_engine_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown ASR engine"):
            utils.create_asr_engine("unknown_model")

    @pytest.mark.parametrize("engine_name,expected_class", [
        ("whisper", Whisper),
        ("canary", Canary),
        ("parakeet", Parakeet),
    ])
    def test_returns_correct_class(self, engine_name, expected_class):
        with patch.object(expected_class, 'download', return_value=None):
            instance = utils.create_asr_engine(engine_name)
            assert isinstance(instance, expected_class)

    def test_cache_returns_same_instance(self):
        with patch.object(Whisper, 'download', return_value=None):
            first = utils.create_asr_engine("whisper", model_size="tiny")
            second = utils.create_asr_engine("whisper", model_size="tiny")
            assert first is second

    def test_different_params_return_different_instances(self):
        with patch.object(Whisper, 'download', return_value=None):
            tiny = utils.create_asr_engine("whisper", model_size="tiny")
            base = utils.create_asr_engine("whisper", model_size="base")
            assert tiny is not base

    def test_different_engines_return_different_instances(self):
        with patch.object(Whisper, 'download', return_value=None):
            with patch.object(Canary, 'download', return_value=None):
                w = utils.create_asr_engine("whisper")
                c = utils.create_asr_engine("canary")
                assert w is not c

    def test_download_called_once_on_first_call(self):
        with patch.object(Whisper, 'download') as mock_download:
            utils.create_asr_engine("whisper", model_size="tiny")
            utils.create_asr_engine("whisper", model_size="tiny")
            mock_download.assert_called_once()

    def test_cache_key_includes_device(self):
        with patch.object(Canary, 'download', return_value=None):
            cpu = utils.create_asr_engine("canary", device="cpu")
            # Different device = different cache key = different instance
            utils._model_cache.clear()
            cpu2 = utils.create_asr_engine("canary", device="cpu")
            assert cpu is not cpu2  # cleared cache so new instance

    def test_empty_engine_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            utils.create_asr_engine("")

    def test_cache_size_grows_with_unique_calls(self):
        with patch.object(Whisper, 'download', return_value=None):
            with patch.object(Canary, 'download', return_value=None):
                utils.create_asr_engine("whisper", model_size="tiny")
                utils.create_asr_engine("whisper", model_size="base")
                utils.create_asr_engine("canary", language="en")
                assert len(utils._model_cache) == 3
