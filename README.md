# ViSEC Speech Emotion Recognition (SER) Enterprise v2.0

> **Đồ án Đột phá / Khóa luận Tốt nghiệp — Đại học Sư phạm Kỹ thuật TP.HCM (HCMUTE)**  
> **Nhóm thực hiện:** Nhóm 2  
> **Thành viên:** Lê Hoàng Triều (MSSV: 24110356) cùng các thành viên Nhóm 2  
> **Tech Stack:** Next.js 16 (React 19) + FastAPI (ASGI) + ONNX Runtime (Wav2Vec2 + ArcFace)  
> **Dữ liệu:** ViSEC (Vietnamese Social Media Emotion Corpus — 4 nhãn: `angry`, `happy`, `neutral`, `sad`)

---

## 1. Sơ Đồ Kiến Trúc Hệ Thống (Architecture)

```text
┌────────────────────────────────────────────────────────┐
│ [Frontend] Next.js 16 (React 19 + Tailwind CSS)        │
│ - Thu âm qua Web Audio API (MIME 'audio/webm'/'wav')   │
│ - Live Canvas Oscilloscope + 4 nút test mẫu nhanh      │
│ - Hiển thị Dominant Emotion, Probability Bars & Latency│
└──────────────────────────┬─────────────────────────────┘
                           │ HTTP POST Multipart (/api/v1/predict/emotion)
                           ▼
┌────────────────────────────────────────────────────────┐
│ [API Gateway] FastAPI + Uvicorn (Zero-Disk I/O)        │
│ - In-Memory Buffer (io.BytesIO), không ghi file ra đĩa │
│ - GET /health: Giám sát CPU %, RAM MB, VRAM GPU MB     │
└──────────────────────────┬─────────────────────────────┘
                           │ Raw Audio Bytes
                           ▼
┌────────────────────────────────────────────────────────┐
│ [DSP Engine] SoundFile + PyDub + NumPy (Zero-Librosa)  │
│ - VAD Silence Trim (top_db = 30) & RMS Check (< 0.002) │
│ - Adaptive Sliding Window: Cửa sổ 3.5s, bước nhảy 2.0s │
└──────────────────────────┬─────────────────────────────┘
                           │ Dynamic Chunks (K, 56000)
                           ▼
┌────────────────────────────────────────────────────────┐
│ [AI Engine] ONNX Runtime Dual-Engine                   │
│ - Tự động dùng CUDA GPU nếu có, tự fallback sang CPU   │
│ - Disaster Recovery: Nạp PyTorch (model_fold_5.pt)     │
│ - Confidence-Weighted Salience Pooling tập hợp kết quả │
└────────────────────────────────────────────────────────┘
```

---

## 2. Điểm Sáng Kỹ Thuật Cốt Lõi

1. **Zero-Librosa Runtime:** Loại bỏ hoàn toàn `librosa`, chuyển sang `soundfile`, `pydub`, `numpy`. Giảm thời gian khởi động backend từ ~4.5s xuống dưới **75ms** (triệt tiêu Cold Start lag).
2. **Adaptive Overlapping Sliding Window:** Cửa sổ 3.5s (56,000 mẫu ở 16kHz), bước nhảy 2.0s (gối đầu 42.8%), kèm chunk vét đuôi, bao phủ **100% thời lượng âm thanh**, không bỏ sót điểm mù thời gian.
3. **Confidence-Weighted Salience Pooling:** Tính toán độ nổi bật cảm xúc từng chunk ($\text{Salience} = \max P - \min P$) và gán trọng số Softmax ($\tau=0.5$), bảo toàn biểu cảm bộc phát (quát giận, cười rộ) thay vì bị làm loãng bởi mean pooling.
4. **Resilient Dual-Engine:** Tự động tăng tốc bằng `CUDAExecutionProvider` nếu có GPU NVIDIA, tự động fallback sang `CPUExecutionProvider` hoặc `PyTorchFallbackEngine` (`model_fold_5.pt`) nếu lỗi file ONNX, đảm bảo không sập server.
5. **Zero-Disk I/O:** Tiếp nhận và giải mã toàn bộ âm thanh (WAV, MP3, WebM/Opus, M4A) trực tiếp trong RAM qua `io.BytesIO`.

---

## 3. Cấu Trúc Thư Mục

```text
emotion-recognition-hcmute/
├── README.md                         # Tài liệu hướng dẫn dự án
├── .gitignore                        # Danh sách loại trừ tệp tạm, cache, model logs
├── model_fold_5.pt                   # Checkpoint PyTorch gốc (Dự phòng Disaster Recovery)
│
├── servers/                          # [Backend] FastAPI & AI Engine
│   ├── requirements.txt              # Thư viện Python runtime
│   ├── app/
│   │   ├── main.py                   # FastAPI app, CORS, routes & system metrics
│   │   ├── schemas.py                # Pydantic models (Health, Predict, Telemetry)
│   │   ├── core/
│   │   │   └── audio_processor.py    # DSP (VAD trim 30dB, RMS check, chunking 3.5s/2.0s)
│   │   └── engine/
│   │       └── onnx_engine.py        # ONNX Singleton Dual-Engine & PyTorch Fallback
│   ├── scripts/
│   │   └── export_onnx.py            # Script trích xuất mô hình ONNX từ PyTorch
│   └── weights/
│       └── ser_model.onnx            # File mô hình ONNX đã tối ưu (372 MB)
│
└── clients/                          # [Frontend] Next.js 16 Web Dashboard
    ├── package.json
    ├── public/
    │   └── samples/                  # 4 mẫu âm thanh benchmark (Angry, Happy, Neutral, Sad)
    └── src/
        ├── app/
        │   ├── page.tsx              # Giao diện chính + Controller chống Race Condition
        │   └── globals.css           # Cấu hình Light Theme
        ├── components/
        │   ├── Header.tsx            # Badge trạng thái GPU/CPU thời gian thực
        │   ├── AudioInput.tsx        # Thu âm Micro, Dropzone & 4 nút test mẫu nhanh
        │   ├── EmotionDashboard.tsx  # Thẻ cảm xúc chính, xác suất 4 nhãn & latency telemetry
        │   └── WaveformVisualizer.tsx# Live Audio Canvas Oscilloscope
        └── hooks/
            └── useAudioRecorder.ts   # Web Audio API hook
```

---

## 4. Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu Cầu Môi Trường:
- **Python:** 3.10 hoặc 3.11
- **Node.js:** 18.x trở lên
- **FFmpeg:** Đã có trong biến môi trường PATH

---

### Bước 1: Khởi Động Backend FastAPI (Terminal 1)
```powershell
# Di chuyển vào thư mục dự án
cd c:\D\XLTN\emotion-recognition-hcmute

# Cài đặt thư viện backend
pip install -r servers/requirements.txt

# Khởi chạy server FastAPI (Port 8000)
python -m uvicorn servers.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Bước 2: Khởi Động Frontend Next.js (Terminal 2)
```powershell
# Chuyển vào thư mục clients
cd c:\D\XLTN\emotion-recognition-hcmute\clients

# Cài đặt node_modules (nếu chạy lần đầu)
npm install

# Khởi chạy Next.js dev server (Port 3000)
npm run dev
```
- **Web App:** Mở trình duyệt tại [http://localhost:3000](http://localhost:3000)

---

## 5. Danh Mục REST API

### 1. `GET /health`
Kiểm tra tình trạng sẵn sàng của mô hình AI và thông số phần cứng.
- **Phản hồi mẫu:**
```json
{
  "status": "online",
  "device": "NVIDIA CUDA GPU",
  "active_provider": "CUDAExecutionProvider",
  "model_loaded": true,
  "system_metrics": {
    "cpu_usage_percent": 12.0,
    "ram_used_mb": 4500.0,
    "ram_total_mb": 16384.0,
    "vram_used_mb": 1420.0,
    "vram_total_mb": 8187.5,
    "gpu_name": "NVIDIA GeForce RTX 4060 Laptop GPU"
  }
}
```

### 2. `POST /api/v1/predict/emotion`
Nhận diện cảm xúc từ file âm thanh tải lên (`multipart/form-data`, trường `file`).
- **Phản hồi thành công:**
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
- **Phản hồi cảnh báo (Quality Gating khi RMS < 0.002 hoặc file < 0.5s):**
```json
{
  "status": "warning",
  "error": "Không phát hiện tiếng người rõ ràng (Nhiễu nền hoặc im lặng, RMS < 0.002).",
  "dominant_emotion": null,
  "dominant_probability": null,
  "probabilities": null,
  "telemetry": {
    "duration": 1.2,
    "rms": 0.0003,
    "num_chunks": 0,
    "dsp_latency_ms": 0.5,
    "inference_latency_ms": 0.0,
    "total_latency_ms": 0.5
  }
}
```

---

## 6. Thông Số Hiệu Năng Thực Nghiệm (Benchmarks)

| Chỉ số | Thông số thực tế |
| :--- | :--- |
| **Mô hình AI Lõi** | Wav2Vec2-Base (13-layer) + BiGRU + SAP + ArcFace |
| **Kích thước mô hình ONNX** | **372 MB** (Đồ thị tối ưu hằng số) |
| **Độ trễ DSP (Tiền xử lý)** | **~1 - 8 ms** (Thuần NumPy & SoundFile) |
| **Độ trễ ONNX GPU (CUDA)** | **~150 - 180 ms** |
| **Độ trễ ONNX CPU** | **~250 - 280 ms** |
| **Tổng trễ End-to-End (E2E)** | **~160 - 290 ms** |
| **Cold Start Lag** | **< 75 ms** (Không dùng librosa) |
| **Độ chính xác (ViSEC 5-Fold)** | **Accuracy: 65.7%** \| **Macro F1: 65.4%** |
