import os
import sys
import io
import shutil

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from fastapi.testclient import TestClient
from servers.app.engine.onnx_engine import ONNXInferenceEngine
from servers.app.engine.pytorch_engine import PyTorchInferenceEngine

def test_engine_fallback_scenario():
    onnx_real_path = os.path.join("servers", "weights", "ser_model.onnx")
    onnx_hidden_path = os.path.join("servers", "weights", "ser_model_hidden.onnx")

    assert os.path.exists(onnx_real_path), f"File ONNX gốc không tồn tại: {onnx_real_path}"

    print("\n" + "=" * 70)
    print("  BẮT ĐẦU KIỂM THỬ KHẢ NĂNG PHỤC HỒI THẢM HỌA (FALLBACK STRESS-TEST)")
    print("=" * 70)

    try:
        # 1. Giả lập mất file / hỏng ổ cứng bằng cách đổi tên file ONNX
        print(f"[1/4] Giả lập mất file ONNX: Đổi tên '{onnx_real_path}' -> '{onnx_hidden_path}'...")
        if os.path.exists(onnx_hidden_path):
            os.remove(onnx_hidden_path)
        shutil.move(onnx_real_path, onnx_hidden_path)
        assert not os.path.exists(onnx_real_path), "File ONNX chưa được ẩn thành công!"

        # Reset singleton engine
        ONNXInferenceEngine.reset_instance()
        PyTorchInferenceEngine.reset_instance()

        # 2. Khởi tạo Engine qua ONNXInferenceEngine.get_instance
        print("[2/4] Khởi tạo Engine khi thiếu file ONNX (Kỳ vọng: KHÔNG CRASH, kích hoạt PyTorch)...")
        engine = ONNXInferenceEngine.get_instance(onnx_real_path)

        assert engine is not None, "Engine bị None sau khi khởi tạo!"
        status = engine.get_status()
        print(f"      --> Trạng thái Engine: {status}")
        print(f"      --> Provider kích hoạt: {status['active_provider']}")
        print(f"      --> Thiết bị: {status['device']}")

        assert "PyTorch" in status["active_provider"] or "PyTorch" in status["device"], \
            f"Engine không chuyển sang PyTorch: {status}"

        # 3. Kiểm thử suy luận trực tiếp với batch 2 chunks (56000 samples)
        print("[3/4] Chạy suy luận batch 2 chunks qua PyTorch Engine Fallback...")
        dummy_chunks = np.random.randn(2, 56000).astype(np.float32)
        probs = engine.predict_chunks(dummy_chunks)

        assert probs.shape == (2, 4), f"Shape kết quả không đúng: {probs.shape}"
        row_sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-5, err_msg="Tổng xác suất không bằng 1.0!")
        print(f"      --> Suy luận thành công: Shape {probs.shape} | Phân phối: {probs[0].round(4)}")

        # 4. Kiểm thử qua API Endpoint FastAPI với TestClient
        print("[4/4] Khởi chạy TestClient FastAPI với Fallback Engine...")
        from servers.app.main import app
        with TestClient(app) as client:
            health_res = client.get("/health")
            assert health_res.status_code == 200, f"Health check thất bại: {health_res.text}"
            health_json = health_res.json()
            print(f"      --> Endpoint /health: HTTP 200 | Body: {health_json}")

            # Thử gọi endpoint phân loại cảm xúc
            samples_dir = os.path.join("clients", "public", "samples")
            sample_wav = os.path.join(samples_dir, "happy_sample.wav")
            if os.path.exists(sample_wav):
                with open(sample_wav, "rb") as f:
                    wav_bytes = f.read()
                pred_res = client.post(
                    "/api/v1/predict/emotion",
                    files={"file": ("happy_sample.wav", wav_bytes, "audio/wav")}
                )
                assert pred_res.status_code == 200, f"Prediction thất bại: {pred_res.text}"
                pred_json = pred_res.json()
                print(f"      --> Endpoint /predict: HTTP 200 | Cảm xúc: {pred_json['dominant_emotion']} ({pred_json['dominant_probability']*100:.1f}%)")

        print("\n" + "=" * 70)
        print("  [SUCCESS] FALLBACK STRESS-TEST HOÀN TẤT: KHÔNG CRASH, HOẠT ĐỘNG HOÀN HẢO!")
        print("=" * 70 + "\n")

    finally:
        # Luôn phục hồi file ONNX gốc bất kể thành công hay thất bại
        if os.path.exists(onnx_hidden_path):
            if os.path.exists(onnx_real_path):
                os.remove(onnx_real_path)
            shutil.move(onnx_hidden_path, onnx_real_path)
            print(f"[*] Đã phục hồi file ONNX gốc: '{onnx_real_path}'")
        ONNXInferenceEngine.reset_instance()
        PyTorchInferenceEngine.reset_instance()

if __name__ == "__main__":
    test_engine_fallback_scenario()
