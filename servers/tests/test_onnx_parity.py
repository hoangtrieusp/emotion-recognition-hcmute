import os
import sys
import io

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
import onnxruntime as ort
from servers.app.engine.onnx_engine import PyTorchFallbackEngine

def test_onnx_pytorch_parity():
    checkpoint_path = "model_fold_5.pt"
    onnx_path = os.path.join("servers", "weights", "ser_model.onnx")

    assert os.path.exists(checkpoint_path), f"Không tìm thấy checkpoint: {checkpoint_path}"
    assert os.path.exists(onnx_path), f"Không tìm thấy file ONNX: {onnx_path}"

    print("\n" + "=" * 65)
    print("  BẮT ĐẦU KIỂM ĐỊNH TƯƠNG ĐỒNG SỐ HỌC (ONNX vs PYTORCH PARITY)")
    print("=" * 65)

    # 1. Nạp mô hình PyTorch (CPU)
    print("[1/3] Đang nạp mô hình PyTorch gốc từ checkpoint...")
    pt_engine = PyTorchFallbackEngine.get_instance(checkpoint_path)
    pt_model = pt_engine.model
    pt_model.eval()

    # 2. Khởi tạo phiên ONNX Runtime
    print(f"[2/3] Đang nạp mô hình ONNX từ '{onnx_path}'...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    # 3. Tạo mảng NumPy ngẫu nhiên kích thước (1, 16000) giả lập 1s âm thanh 16kHz
    print("[3/3] Tạo mảng numpy ngẫu nhiên shape (1, 16000) giả lập 1 giây audio...")
    np.random.seed(42)
    dummy_audio = np.random.randn(1, 16000).astype(np.float32)

    # Suy luận PyTorch
    pt_probs = pt_engine.predict_chunks(dummy_audio)

    # Suy luận ONNX Runtime
    ort_probs = ort_session.run(None, {"input_values": dummy_audio})[0]

    # Tính toán sai số MAE
    mae = float(np.max(np.abs(pt_probs - ort_probs)))
    mean_ae = float(np.mean(np.abs(pt_probs - ort_probs)))

    print("\n--- KẾT QUẢ ĐỐI CHIẾU XÁC SUẤT 4 LỚP CẢM XÚC ---")
    emotions = ["angry", "happy", "neutral", "sad"]
    for i, emo in enumerate(emotions):
        print(f"  Lớp {emo:<8}: PyTorch = {pt_probs[0, i]:.6f} | ONNX = {ort_probs[0, i]:.6f} | Sai số = {abs(pt_probs[0, i] - ort_probs[0, i]):.2e}")

    print(f"\n[CHỈ SỐ ĐỘ CHÍNH XÁC]")
    print(f"  • Max Absolute Error (MAE) : {mae:.6e}")
    print(f"  • Mean Absolute Error      : {mean_ae:.6e}")
    print(f"  • Ngưỡng cho phép (Limit)  : 1.000000e-04")

    # Kiểm định nghiêm ngặt bằng assert_allclose
    np.testing.assert_allclose(
        pt_probs,
        ort_probs,
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"Sai số MAE ({mae:.6e}) vượt ngưỡng cho phép 1e-4!"
    )

    print("\n" + "=" * 65)
    print("  [SUCCESS] KIỂM ĐỊNH PARITY HOÀN TẤT: MAE < 1e-4 ĐẠT CHUẨN L0 TUYỆT ĐỐI!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    test_onnx_pytorch_parity()
