from whisper import Whisper
from parakeet import Parakeet
from canary import Canary

MODEL_INSTANCE = None
MODEL_NAME = None

def create_asr_engine(engine_name: str, device: str = "cpu", **kwargs):
    global MODEL_INSTANCE, MODEL_NAME

    if MODEL_INSTANCE is not None and MODEL_NAME == engine_name:
        return MODEL_INSTANCE

    import gc
    import torch

    MODEL_INSTANCE = None
    gc.collect()
    torch.cuda.empty_cache()

    engines = {
        "whisper": Whisper,
        "parakeet": Parakeet,
        "canary": Canary,
    }

    model = engines[engine_name](device=device, **kwargs)

    MODEL_INSTANCE = model
    MODEL_NAME = engine_name

    return model
    
