# ViSEC Speech Emotion Recognition (SER)

> **Kiến trúc:** Next.js 16 (App Router, Turbopack) + FastAPI (ASGI) + ONNX Runtime (CUDA / CPU Fallback) + Adaptive DSP Chunking
> **Dataset:** ViSEC (Vietnamese Social Media Emotion Corpus)

---

## 1. Giới thiệu Tổng quan

Hệ thống nhận diện cảm xúc giọng nói tiếng Việt được nâng cấp toàn diện từ mô hình thử nghiệm giữa kỳ (Gradio đơn luồng) thành kiến trúc **Microservices chuẩn Production**:

- **4 Cảm xúc nhận diện (ViSEC 4 classes):**
  - 🔥 **Tức giận (Angry)**
  - 😄 **Vui vẻ (Happy)**
  - 😐 **Bình thường (Neutral)**
  - 😢 **Buồn bã (Sad)**
- **Kiến trúc Mô hình AI Lõi:**
  $$\text{Wav2Vec2-Base (13-layer Weighted Pooling)} \longrightarrow \text{BiGRU} \longrightarrow \text{Self-Attention Pooling (Mean + Std)} \longrightarrow \text{ArcFace Cosine Head}$$
- **Tối ưu hóa Suy luận:**
  - Toàn bộ mô hình PyTorch được xuất sang định dạng tĩnh **ONNX** (372 MB) với trục kích thước động (`dynamic_axes`), tối ưu hóa đồ thị bằng `onnxruntime`.
  - Tự động nhận diện phần cứng: Tận dụng **NVIDIA CUDA GPU** và tự động chuyển về **CPU** trên máy không có GPU rời.
  - Tích hợp cơ chế **Phục hồi Thảm họa (Disaster Recovery)**: Tự động chuyển đổi mượt mà sang `PyTorchFallbackEngine` (`model_fold_5.pt`) nếu file ONNX gặp sự cố, đảm bảo hệ thống không bao giờ bị gián đoạn.
- **Độ trễ toàn hệ thống (E2E Latency):** **~150ms** trên GPU và **~200 - 300ms** trên CPU, loại bỏ hoàn toàn độ trễ khởi động của thư viện nặng.

---

## 2. Bản vẽ Kiến trúc Hệ thống (L0 - L5)

```text
┌────────────────────────────────────────────────────────────────────────┐
│ [L5] Presentation Layer: Next.js 16 + React 19 + Tailwind CSS v4       │
│ - Micro Recorder (Web Audio API oscilloscope, Blob 'audio/webm'/'wav') │
│ - 4 Benchmark Preset Buttons + Chống Race Condition (isAnalyzing Lock) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP Multipart POST (/api/v1/predict/emotion)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ [L4] API Gateway: FastAPI + Uvicorn ASGI                               │
│ - CORS Middleware bảo vệ đa cổng (localhost:3000, regex wildcard)      │
│ - Async Non-blocking I/O Stream & Centralized Error Handler (400/500)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Raw Audio Bytes
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ [L1-L2] Vectorized DSP Engine: Pure NumPy / SoundFile / PyDub          │
│ - Zero Librosa Cold Start (Khởi động API trong < 80ms)                │
│ - VAD Silence Trim 30dB + RMS Quality Gating (Loại bỏ âm rỗng RMS<0.002)│
│ - Adaptive Overlapping Chunking (Window 3.5s, Hop 2.0s - 100% bao phủ) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Dynamic Batch Chunks (K, 56000)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ [L0] High-Performance Inference Engine: ONNX Runtime                   │
│ - Hardware Detection: CUDAExecutionProvider -> CPUExecutionProvider   │
│ - Disaster Recovery: Tự động fallback sang PyTorch (model_fold_5.pt)   │
│ - Confidence-Weighted Salience Pooling: Bảo toàn cảm xúc bộc phát      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu trúc Mã Nguồn Dự Án (Project Structure)

```text
emotion-recognition-hcmute/
├── README.md                         # Tài liệu hướng dẫn tổng quan dự án
├── .gitignore                        # Cấu hình bỏ qua cache, build, và checkpoints
├── model_fold_5.pt                   # Checkpoint PyTorch gốc (dự phòng cho Fallback Engine)
│
├── servers/                          # [Backend] FastAPI & AI Inference Engine
│   ├── requirements.txt              # Danh mục đầy đủ thư viện Python
│   ├── __init__.py
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI ASGI App, Endpoints & CORS
│   │   ├── schemas.py                # Pydantic v2 Models & Telemetry Schema
│   │   ├── core/
│   │   │   └── audio_processor.py    # DSP Preprocessing (VAD Trim, RMS, Chunking)
│   │   └── engine/
│   │       └── onnx_engine.py        # Hợp nhất ONNX Runtime + PyTorch Fallback Engine
│   ├── scripts/
│   │   └── export_onnx.py            # Script xuất model ONNX chính thức từ checkpoint
│   └── weights/
│       └── ser_model.onnx            # File mô hình ONNX chuẩn hóa (372 MB)
│
└── clients/                          # [Frontend] Next.js 16 Web Dashboard
    ├── package.json
    ├── tsconfig.json
    ├── next.config.ts
    ├── README.md                     # Tài liệu hướng dẫn chi tiết Frontend
    ├── public/
    │   └── samples/                  # 4 benchmark sample files cho nút Test Nhanh
    └── src/
        ├── app/
        │   ├── layout.tsx
        │   ├── page.tsx              # Dashboard View + isAnalyzing Lock
        │   └── globals.css           # Styling
        ├── components/
        │   ├── Header.tsx            # Real-time System/GPU Status Badge
        │   ├── WaveformVisualizer.tsx# Live Micro Oscilloscope Canvas
        │   ├── AudioInput.tsx        # Mic Recorder, Dropzone & Preset Buttons
        │   └── EmotionDashboard.tsx  # Dominant Emotion Card & Softmax Progress Bars
        └── hooks/
            └── useAudioRecorder.ts   # Web Audio API hook (Strict audio/webm Blob)
```

---

## 4. Hướng dẫn Khởi chạy Hệ thống

### Yêu cầu Tiên quyết:
- **Python:** 3.10 hoặc 3.11
- **Node.js:** 18.x hoặc 20.x
- **FFmpeg:** Đã cài đặt trên hệ thống (hoặc nằm trong biến môi trường PATH)

---

### Bước 1: Khởi động Backend FastAPI (Port 8000)
Mở **Terminal 1** tại thư mục gốc của dự án:

```powershell
# 1. Cài đặt toàn bộ thư viện backend
pip install -r servers/requirements.txt

# 2. Khởi chạy server FastAPI tại cổng 8000
python -m uvicorn servers.app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Bước 2: Khởi động Frontend Next.js (Port 3000)
Mở **Terminal 2** tại thư mục `clients/`:

```powershell
cd clients

# 1. Cài đặt dependencies (nếu clone lần đầu)
npm install

# 2. Khởi chạy giao diện phát triển
npm run dev
```

- Mở trình duyệt tại: [http://localhost:3000](http://localhost:3000)

---

## 5. Danh mục REST API Chính

### 1. `GET /health`
Kiểm tra trạng thái sẵn sàng của mô hình và phần cứng thực thi.
- **Response mẫu:**
  ```json
  {
    "status": "online",
    "device": "NVIDIA CUDA GPU",
    "active_provider": "CUDAExecutionProvider",
    "model_loaded": true
  }
  ```

### 2. `POST /api/v1/predict/emotion`
Nhận diện cảm xúc từ file âm thanh tải lên (WAV, MP3, WebM/Opus, M4A, OGG, FLAC).
- **Request:** `multipart/form-data` với trường `file`.
- **Response mẫu:**
  ```json
  {
    "status": "success",
    "dominant_emotion": "happy",
    "dominant_probability": 0.9893,
    "probabilities": {
      "angry": 0.0015,
      "happy": 0.9893,
      "neutral": 0.0076,
      "sad": 0.0016
    },
    "telemetry": {
      "duration": 2.8,
      "rms": 0.37821,
      "num_chunks": 1,
      "dsp_latency_ms": 1.15,
      "inference_latency_ms": 175.24,
      "total_latency_ms": 176.39
    },
    "error": null
  }
  ```

---

## 6. Các Điểm Sáng Kỹ thuật Dành Cho Báo Cáo Hội Đồng

1. **Khắc phục Triệt để Điểm Mù Thời Gian (Temporal Blind Spot):**
   - Thay vì cắt thô 2 đầu của file dài như bản demo giữa kỳ, hệ thống áp dụng **Adaptive Overlapping Sliding Window** (cửa sổ 3.5s, bước nhảy 2.0s), đảm bảo bao phủ 100% thời lượng câu nói.
   - Kết hợp **Confidence-Weighted Salience Pooling** để bảo toàn các cảm xúc bộc phát ngắn hạn (tiếng quát giận, tiếng cười) không bị loãng bởi các đoạn nói đều đều.

2. **Tối ưu Hóa Khởi Động & Tiền Xử Lý (Zero Librosa Runtime):**
   - Loại bỏ việc nạp `librosa` trên FastAPI runtime giúp triệt tiêu hoàn toàn Cold Start lag (thời gian khởi động backend giảm từ 4s xuống dưới 80ms).
   - Giải mã âm thanh đa định dạng (đặc biệt là WebM/Opus từ trình duyệt) về chuẩn PCM 16kHz Float32 bằng SoundFile & PyDub.

3. **Cơ Chế Phục Hồi Thảm Họa (Disaster Recovery):**
   - Tự động nhận diện phần cứng: Ưu tiên `CUDAExecutionProvider` nếu có GPU NVIDIA, tự động chuyển về `CPUExecutionProvider` nếu chạy trên laptop trình chiếu không card rời.
   - Nếu tệp mô hình `.onnx` bị lỗi hoặc biến mất, hệ thống kích hoạt `PyTorchFallbackEngine` từ `model_fold_5.pt` mà **không bị crash server**.

4. **Trực Quan Hóa & Khả năng Quan Sát (Observability):**
   - Dashboard hiển thị phân tách chi tiết độ trễ: **DSP Latency (ms)**, **ONNX Inference Latency (ms)** và **Total E2E (ms)**.
   - Giao diện Sáng (Clean Light Theme) với độ tương phản cao, tối ưu hiển thị trên màn chiếu của hội đồng bảo vệ.
