import os
import sys
import io

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from servers.app.main import app

def test_health_and_predict():
    with TestClient(app) as client:
        # 1. Test /health
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        data_health = res_health.json()
        print(f"[TEST PASS] Health check response: {data_health}")
        assert data_health["model_loaded"] is True

        # 2. Tạo file WAV hợp lệ (âm thanh 3 giây, 16kHz)
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Giả lập tín hiệu giọng nói (sóng mang 220Hz điều chế tần số)
        audio_data = 0.5 * np.sin(2 * np.pi * 220 * t) * (1.0 + 0.3 * np.sin(2 * np.pi * 5 * t))
        audio_data = audio_data.astype(np.float32)

        buf = io.BytesIO()
        sf.write(buf, audio_data, sr, format="WAV")
        buf.seek(0)

        # 3. Test POST /api/v1/predict/emotion
        response = client.post(
            "/api/v1/predict/emotion",
            files={"file": ("test_speech.wav", buf.read(), "audio/wav")}
        )
        assert response.status_code == 200, f"Prediction failed: {response.text}"
        result = response.json()
        print(f"[TEST PASS] Prediction response: {result}")
        assert result["status"] == "success"
        assert result["dominant_emotion"] in ["angry", "happy", "neutral", "sad"]
        assert result["dominant_probability"] > 0
        assert "telemetry" in result
        print(f"--> Inference Latency: {result['telemetry']['inference_latency_ms']} ms")
        print(f"--> Total E2E Latency: {result['telemetry']['total_latency_ms']} ms")

if __name__ == "__main__":
    test_health_and_predict()
