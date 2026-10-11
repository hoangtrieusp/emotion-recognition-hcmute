# ViSEC SER Enterprise v2.0 (Lean Edition) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai hệ thống Web Nhận diện Cảm xúc Tiếng nói tinh gọn (Next.js Light Theme + FastAPI + ONNX Runtime CUDA/CPU) thay thế Gradio, phục vụ báo cáo Đồ án Cuối kỳ.

**Architecture:** 2 Tiers rõ ràng:
- Frontend (`clients/`): Next.js Light Theme, Audio Waveform Recorder, Drag-drop Upload, 4 nút Sample Preset, Emotion Dashboard.
- Backend (`servers/`): FastAPI, ONNX Runtime (CUDA/CPU fallback), DSP Audio Processor (SoundFile/PyDub decode, VAD trim, Overlapping Chunking, Salience Pooling).

**Tech Stack:** Next.js, React, Tailwind CSS, Web Audio API, FastAPI, Uvicorn, ONNX Runtime, NumPy, SoundFile, PyDub.

**Spec:** `docs/superpowers/specs/2026-10-11-ser-enterprise-system-design.md`

## Global Constraints
- Target Audio Format: 16000Hz, Mono, Float32.
- 4 Emotion Classes: `{0: "angry", 1: "happy", 2: "neutral", 3: "sad"}`.
- Zero Librosa on FastAPI Runtime: Không dùng librosa để tránh lag khởi động.
- Theme: Giao diện sáng (Light Theme) chuyên nghiệp, tối ưu hiển thị máy chiếu.
- Resilience: Tự động fallback sang CPU nếu không có CUDA GPU.

---

### Task 1: Xuất Model ONNX từ Checkpoint (L0)
- [ ] **Step 1:** Viết script `servers/scripts/export_onnx.py` đóng gói `SERModel` + `ArcFaceInferenceHead` từ `model_fold_5.pt` sang `servers/weights/ser_model.onnx` với `dynamic_axes = {'input_values': {0: 'batch_size', 1: 'sequence_length'}, 'logits': {0: 'batch_size'}}`.
- [ ] **Step 2:** Chạy script xuất file `.onnx` và kiểm tra sai số số học $MAE < 10^{-4}$ so với PyTorch.
- [ ] **Step 3:** Commit model exporter script.

### Task 2: Xây dựng Lõi DSP Audio Processor (L1, L2)
- [ ] **Step 1:** Tạo `servers/app/core/audio_processor.py` hỗ trợ giải mã đa định dạng (WAV, MP3, WebM/Opus) sang Float32 16kHz.
- [ ] **Step 2:** Tích hợp Silence Trimming (top_db=30), RMS Energy check (< 0.002), Adaptive Overlapping Chunking (3.5s window, 2.0s hop), và Confidence-Weighted Salience Pooling.
- [ ] **Step 3:** Viết unit test nhanh xác nhận xử lý audio thành công.
- [ ] **Step 4:** Commit audio processor.

### Task 3: Xây dựng Backend FastAPI & ONNX Engine (L0, L4)
- [ ] **Step 1:** Tạo `servers/app/engine/onnx_engine.py` (Singleton session nạp `ser_model.onnx`, hỗ trợ CUDA/CPU provider, warm-up test).
- [ ] **Step 2:** Tạo `servers/app/schemas.py` và `servers/app/main.py` với 2 endpoint: `GET /health` và `POST /api/v1/predict/emotion`.
- [ ] **Step 3:** Cài đặt dependencies `servers/requirements.txt` và chạy test API với 1 file audio mẫu.
- [ ] **Step 4:** Commit backend code.

### Task 4: Xây dựng Frontend Next.js (Light Theme) (L5)
- [ ] **Step 1:** Khởi tạo Next.js App trong `clients/` với Tailwind CSS và Lucide React.
- [ ] **Step 2:** Xây dựng Header (Logo, Tên đề tài, Trạng thái GPU/CPU).
- [ ] **Step 3:** Xây dựng `AudioInput.tsx` tích hợp Thu âm mic (Web Audio API oscilloscope), Kéo thả file, và 4 nút Test mẫu nhanh.
- [ ] **Step 4:** Xây dựng `EmotionDashboard.tsx` hiển thị Cảm xúc chính, Progress bars 4 nhãn, và Thẻ chỉ số kỹ thuật (Duration, RMS, Latency ms).
- [ ] **Step 5:** Kiểm tra build giao diện Next.js.
- [ ] **Step 6:** Commit frontend code.

### Task 5: Tích hợp Toàn diện & Kiểm thử (Verification)
- [ ] **Step 1:** Khởi động Backend FastAPI và Frontend Next.js.
- [ ] **Step 2:** Thực hiện test thu mic và test cả 4 nhãn cảm xúc bằng sample files.
- [ ] **Step 3:** Đo độ trễ E2E và hoàn thiện hướng dẫn chạy hệ thống.
