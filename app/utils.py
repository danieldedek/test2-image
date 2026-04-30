from whisper import Whisper
from parakeet import Parakeet
from canary import Canary
import torch
import gc

current_model = None
current_engine = None


def create_asr_engine(engine_name: str, device: str = "cpu", **kwargs):
    global current_model, current_engine

    if current_engine != engine_name:
        if current_model is not None:
            del current_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        current_model = None

    if current_model is None:
        engines = {
            "whisper": Whisper,
            "parakeet": Parakeet,
            "canary": Canary,
        }

        if engine_name not in engines:
            raise ValueError(f"Unknown ASR engine: {engine_name}")

        current_model = engines[engine_name](device=device, **kwargs)
        current_engine = engine_name

    return current_model
    
