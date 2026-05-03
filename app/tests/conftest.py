import os
import sys
import struct
import math
import wave
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def _make_sine_wav(path: str, duration_seconds: float, sample_rate: int = 16000, freq: float = 440.0):
    n_samples = int(sample_rate * duration_seconds)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        frames = []
        for i in range(n_samples):
            sample = int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.append(struct.pack('<h', sample))
        wf.writeframes(b''.join(frames))


@pytest.fixture(scope="session")
def short_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "short.wav"
    _make_sine_wav(str(p), duration_seconds=3.0)
    return str(p)


@pytest.fixture(scope="session")
def medium_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "medium.wav"
    _make_sine_wav(str(p), duration_seconds=35.0)
    return str(p)


@pytest.fixture(scope="session")
def long_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "long.wav"
    _make_sine_wav(str(p), duration_seconds=90.0)
    return str(p)


@pytest.fixture(scope="session")
def silent_wav(tmp_path_factory):
    p = tmp_path_factory.mktemp("audio") / "silent.wav"
    n_samples = 16000 * 3
    with wave.open(str(p), 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b'\x00' * n_samples * 2)
    return str(p)


@pytest.fixture(scope="session")
def real_wav_files():
    try:
        from test_config import USE_REAL_FILES, MAX_REAL_FILE_DURATION
    except ImportError:
        return []

    if not USE_REAL_FILES:
        return []

    upload_dir = "/app/uploads"
    if not os.path.isdir(upload_dir):
        return []

    results = []
    for fname in os.listdir(upload_dir):
        if not fname.lower().endswith(".wav"):
            continue
        fpath = os.path.join(upload_dir, fname)
        try:
            with wave.open(fpath, 'r') as wf:
                duration = wf.getnframes() / wf.getframerate()
            if duration <= MAX_REAL_FILE_DURATION:
                results.append(fpath)
        except Exception:
            pass
    return results


@pytest.fixture(scope="session")
def flask_client():
    sys.path.insert(0, '/app')
    from main import app
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    with app.test_client() as client:
        yield client
