from whisper import Whisper
from parakeet import Parakeet
from canary import Canary


def create_asr_engine(engine_name: str, device: str = "cpu"):
    engines = {
        "whisper": Whisper,
        "parakeet": Parakeet,
        "canary": Canary,
    }

    if engine_name not in engines:
        raise ValueError(f"Unknown ASR engine: {engine_name}")

    return engines[engine_name](device=device)
