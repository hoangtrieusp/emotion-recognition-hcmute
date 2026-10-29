import io
import os
import glob
import pytest
import numpy as np
import soundfile as sf
import scipy.signal as signal
from fastapi.testclient import TestClient
try:
    from servers.app.main import app
    from servers.app.core.audio_processor import preprocess_audio_pipeline, aggregate_salience_probabilities
    from servers.app.engine.onnx_engine import ONNXInferenceEngine
except ImportError:
    from app.main import app
    from app.core.audio_processor import preprocess_audio_pipeline, aggregate_salience_probabilities
    from app.engine.onnx_engine import ONNXInferenceEngine

EMOTIONS = ["angry", "happy", "neutral", "sad"]
SAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "clients", "public", "samples"))
VLC_MP3_PATH = r"c:\D\XLTN\mp3\vlc-record-2026-10-29-15h57m32s-full_audio.mp3-.mp3"

@pytest.fixture(scope="session")
def engine():
    return ONNXInferenceEngine.get_instance()

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

def _to_wav_bytes(sig: np.ndarray, sr: int = 16000) -> bytes:
    bio = io.BytesIO()
    sf.write(bio, sig.astype(np.float32), sr, format="WAV")
    return bio.getvalue()

def test_clean_speech_all_emotions(engine):
    targets = {
        "angry_sample.wav": "angry",
        "happy_sample.wav": "happy",
        "neutral_sample.wav": "neutral",
        "sad_sample.wav": "sad",
    }
    for filename, expected_emotion in targets.items():
        filepath = os.path.join(SAMPLES_DIR, filename)
        assert os.path.exists(filepath), f"File {filepath} not found"
        with open(filepath, "rb") as f:
            data = f.read()
        res = preprocess_audio_pipeline(data)
        assert res["is_valid"] is True, f"Clean sample {filename} was rejected: {res.get('error')}"
        probs = engine.predict_chunks(res["chunks"])
        final_probs = aggregate_salience_probabilities(probs, res.get("chunk_weights"))
        predicted_idx = int(np.argmax(final_probs))
        predicted_emotion = EMOTIONS[predicted_idx]
        confidence = float(final_probs[predicted_idx])
        assert predicted_emotion == expected_emotion, f"{filename} expected {expected_emotion}, got {predicted_emotion} ({confidence:.4f})"
        assert confidence >= 0.85, f"{filename} confidence {confidence:.4f} is below 0.85"

def test_speech_with_ambient_noise(engine):
    filepath = os.path.join(SAMPLES_DIR, "happy_sample.wav")
    sig, sr = sf.read(filepath, dtype="float32")
    sig_rms = np.sqrt(np.mean(sig ** 2))
    target_noise_rms = sig_rms / (10 ** (12 / 20))
    gaussian_noise = np.random.normal(0, target_noise_rms, len(sig)).astype(np.float32)
    t = np.linspace(0, len(sig) / sr, len(sig), endpoint=False)
    hum_60hz = (target_noise_rms * 0.5) * np.sin(2 * np.pi * 60 * t).astype(np.float32)
    noisy_sig = sig + gaussian_noise + hum_60hz
    wav_bytes = _to_wav_bytes(noisy_sig, sr)
    res = preprocess_audio_pipeline(wav_bytes)
    assert res["is_valid"] is True
    probs = engine.predict_chunks(res["chunks"])
    final_probs = aggregate_salience_probabilities(probs, res.get("chunk_weights"))
    predicted_emotion = EMOTIONS[int(np.argmax(final_probs))]
    assert predicted_emotion == "happy"
    assert final_probs[EMOTIONS.index("happy")] >= 0.70

def test_speech_with_background_music_beats(engine):
    filepath = os.path.join(SAMPLES_DIR, "neutral_sample.wav")
    sig, sr = sf.read(filepath, dtype="float32")
    beat = np.zeros_like(sig)
    for t_sec in np.arange(0.2, len(sig) / sr, 0.4):
        idx = int(t_sec * sr)
        pulse_len = int(0.04 * sr)
        if idx + pulse_len < len(sig):
            t_pulse = np.linspace(0, 0.04, pulse_len, endpoint=False)
            beat[idx : idx + pulse_len] += 0.06 * np.sin(2 * np.pi * 90 * t_pulse) * np.exp(-t_pulse * 80)
    mixed = sig + beat
    wav_bytes = _to_wav_bytes(mixed, sr)
    res = preprocess_audio_pipeline(wav_bytes)
    assert res["is_valid"] is True
    probs = engine.predict_chunks(res["chunks"])
    final_probs = aggregate_salience_probabilities(probs, res.get("chunk_weights"))
    predicted_emotion = EMOTIONS[int(np.argmax(final_probs))]
    assert predicted_emotion == "neutral"
    assert final_probs[EMOTIONS.index("angry")] < 0.20

def test_non_speech_rejection():
    if os.path.exists(VLC_MP3_PATH):
        with open(VLC_MP3_PATH, "rb") as f:
            vlc_data = f.read()
        res = preprocess_audio_pipeline(vlc_data)
        assert res["is_valid"] is False, "EDM instrumental track must be rejected as non-speech"
        assert "giọng nói" in res["error"] or "tiếng" in res["error"]
    sr = 16000
    white_noise = np.random.normal(0, 0.05, sr * 3).astype(np.float32)
    res_white = preprocess_audio_pipeline(_to_wav_bytes(white_noise, sr))
    assert res_white["is_valid"] is False, "Pure white noise must be rejected as non-speech"
    hum_60 = (0.1 * np.sin(2 * np.pi * 60 * np.linspace(0, 3, sr * 3))).astype(np.float32)
    res_hum = preprocess_audio_pipeline(_to_wav_bytes(hum_60, sr))
    assert res_hum["is_valid"] is False, "Pure electrical hum must be rejected as non-speech"
    synth_880 = (0.1 * signal.sawtooth(2 * np.pi * 880 * np.linspace(0, 3, sr * 3))).astype(np.float32)
    res_synth = preprocess_audio_pipeline(_to_wav_bytes(synth_880, sr))
    assert res_synth["is_valid"] is False, "Pure high synth tone must be rejected as non-speech"

def test_short_and_silent_audio_rejection():
    sr = 16000
    short_sig = np.random.normal(0, 0.05, int(sr * 0.3)).astype(np.float32)
    res_short = preprocess_audio_pipeline(_to_wav_bytes(short_sig, sr))
    assert res_short["is_valid"] is False
    assert "quá ngắn" in res_short["error"]
    silent_sig = np.zeros(sr * 2, dtype=np.float32)
    res_silent = preprocess_audio_pipeline(_to_wav_bytes(silent_sig, sr))
    assert res_silent["is_valid"] is False
    assert "RMS < 0.002" in res_silent["error"]

def test_endpoint_api_integration(client):
    clean_path = os.path.join(SAMPLES_DIR, "happy_sample.wav")
    with open(clean_path, "rb") as f:
        response = client.post("/api/v1/predict/emotion", files={"file": ("happy.wav", f, "audio/wav")})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["dominant_emotion"] == "happy"
    assert data["probabilities"]["happy"] >= 0.85
    sr = 16000
    white_noise = np.random.normal(0, 0.05, sr * 3).astype(np.float32)
    white_bytes = _to_wav_bytes(white_noise, sr)
    resp_noise = client.post("/api/v1/predict/emotion", files={"file": ("noise.wav", io.BytesIO(white_bytes), "audio/wav")})
    assert resp_noise.status_code == 200
    data_noise = resp_noise.json()
    assert data_noise["status"] == "warning"
    assert data_noise["dominant_emotion"] is None
    assert "Không phát hiện" in data_noise["error"]
