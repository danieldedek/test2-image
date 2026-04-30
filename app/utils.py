_model_cache = {}
_current_key = None

def create_asr_engine(engine_name: str, device: str = "cpu", **kwargs):
    global _current_key

    key = (engine_name, device, tuple(sorted(kwargs.items())))

    if key == _current_key:
        return _model_cache[key]

    _model_cache.clear()

    engines = {
        "whisper": Whisper,
        "parakeet": Parakeet,
        "canary": Canary,
    }

    model = engines[engine_name](device=device, **kwargs)
    model.download()

    _model_cache[key] = model
    _current_key = key

    return model
    
