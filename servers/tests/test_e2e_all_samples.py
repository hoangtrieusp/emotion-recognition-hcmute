import os
import sys
import io

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient
from servers.app.main import app

def test_all_samples_e2e():
    samples_dir = os.path.join("clients", "public", "samples")
    sample_files = [
        "angry_sample.wav",
        "happy_sample.wav",
        "neutral_sample.wav",
        "sad_sample.wav"
    ]

    with TestClient(app) as client:
        print("\n" + "="*50)
        print("  BẮT ĐẦU KIỂM ĐỊNH TOÀN DIỆN 4 FILE MẪU E2E")
        print("="*50)

        for filename in sample_files:
            filepath = os.path.join(samples_dir, filename)
            assert os.path.exists(filepath), f"Missing sample file: {filepath}"

            with open(filepath, "rb") as f:
                content = f.read()

            response = client.post(
                "/api/v1/predict/emotion",
                files={"file": (filename, content, "audio/wav")}
            )
            assert response.status_code == 200, f"Failed on {filename}: {response.text}"
            res_json = response.json()
            assert res_json["status"] == "success", f"Prediction status not success on {filename}"

            dom_emo = res_json["dominant_emotion"]
            dom_prob = res_json["dominant_probability"]
            telemetry = res_json["telemetry"]

            print(f"\n[FILE: {filename}]")
            print(f"  -> Cảm xúc dự đoán: {dom_emo.upper()} ({dom_prob * 100:.1f}%)")
            print(f"  -> Chi tiết xác suất: {res_json['probabilities']}")
            print(f"  -> Thời lượng: {telemetry['duration']}s | RMS: {telemetry['rms']} | Chunks: {telemetry['num_chunks']}")
            print(f"  -> DSP: {telemetry['dsp_latency_ms']} ms | ONNX Inference: {telemetry['inference_latency_ms']} ms | Tổng E2E: {telemetry['total_latency_ms']} ms")

        print("\n" + "="*50)
        print("  [SUCCESS] 100% CẢ 4 FILE MẪU CHẠY THÀNH CÔNG VỚI ĐỘ TRỄ SIÊU TỐC!")
        print("="*50)

if __name__ == "__main__":
    test_all_samples_e2e()
