# HỆ THỐNG NHẬN DIỆN CẢM XÚC TIẾNG NÓI (SER ENTERPRISE V2.0 - TINH GỌN)
## ARCHITECTURE & TECHNICAL SPECIFICATION

---

### 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)
- **Tên dự án:** ViSEC Speech Emotion Recognition Web System
- **Mục tiêu:** Chuyển đổi mô hình nghiên cứu giữa kỳ (Wav2Vec2 + BiGRU + SAP + ArcFace Head, checkpoint `model_fold_5.pt`, 4 nhãn: `angry`, `happy`, `neutral`, `sad`) từ Gradio sang một hệ thống Web hoàn chỉnh (Next.js + FastAPI + ONNX Runtime) phục vụ báo cáo Đồ án Cuối kỳ.
- **Tiêu chí thiết kế:** Tinh gọn, thực dụng, giao diện sáng (Light Theme) chuyên nghiệp để chiếu máy chiếu, không tính năng thừa (bỏ MCP, bỏ Session History, bỏ DB, bỏ stress test).

---

### 2. BẢN VẼ KIẾN TRÚC HỆ THỐNG

```
===================================================================================
[FRONTEND] : NEXT.JS / REACT (Clean Light Theme, Tailwind CSS)
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Micro Live Recorder (Web Audio API) + Waveform Visualizer sóng âm thời gian thực│
 │ + Audio Upload Dropzone (Kéo thả .wav, .mp3, .m4a, .webm)                     │
 │ + Quick Test Presets (4 nút bấm test nhanh: Vui, Buồn, Giận, Bình thường)    │
 │ + Emotion Dashboard:                                                          │
 │   - Dominant Emotion Card (Biểu tượng & Màu sắc trực quan)                   │
 │   - Confidence Progress Bars (Tỷ lệ phần trăm 4 cảm xúc)                     │
 │   - Metric Card: Duration (s), RMS Energy, Độ trễ Latency (ms)               │
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ HTTP POST /multipart (Audio File)
                                        ▼
===================================================================================
[BACKEND GATEWAY] : FASTAPI (Async Uvicorn)
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Endpoint `GET  /health`                : Kiểm tra trạng thái GPU/CPU         │
 │ + Endpoint `POST /api/v1/predict/emotion`: Nhận file âm thanh, trả kết quả JSON│
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ Raw Audio Bytes
                                        ▼
===================================================================================
[DSP PRE-PROCESSING] : SOUNDFILE / PYDUB / NUMPY
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Giải mã mọi định dạng (WAV, MP3, WebM/Opus) sang Float32 16kHz Mono         │
 │ + Cắt khoảng lặng tĩnh 2 đầu (Silence Trimming top_db=30)                     │
 │ + Kiểm tra RMS Energy: RMS < 0.002 hoặc Duration < 0.5s -> Cảnh báo âm thanh  │
 │ + Adaptive Overlapping Window (> 3.5s -> Cửa sổ trượt 3.5s, stride 2.0s)      │
 │ + Confidence-Weighted Salience Pooling (Tránh loãng cảm xúc cao trào)         │
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ Normalized Float32 Array
                                        ▼
===================================================================================
[AI INFERENCE ENGINE] : ONNX RUNTIME (CUDA -> CPU Fallback)
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Singleton Session nạp `ser_model.onnx` lúc khởi động                        │
 │ + Hybrid Provider: CUDAExecutionProvider -> Fallback CPUExecutionProvider    │
 │ + Single-pass Dynamic Batch Inference -> Softmax Probabilities                │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. CẤU TRÚC MÃ NGUỒN TINH GỌN

```
emotion-recognition-hcmute/
├── clients/                       # Next.js Frontend
│   ├── public/
│   │   └── samples/               # 4 file âm thanh mẫu (vui, buồn, giận, bình thường)
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx         # Root Layout (Light Theme thanh lịch)
│   │   │   ├── page.tsx           # Trang chính
│   │   │   └── globals.css        # Tailwind styling
│   │   └── components/
│   │       ├── Header.tsx         # Thanh tiêu đề & trạng thái kết nối GPU/CPU
│   │       ├── AudioInput.tsx     # Thu mic + Upload file + Test mẫu 4 cảm xúc
│   │       ├── WaveformVisualizer.tsx # Hiển thị sóng âm khi ghi âm
│   │       └── EmotionDashboard.tsx   # Kết quả cảm xúc & chỉ số độ trễ ms
│   ├── package.json
│   └── tailwind.config.ts
│
├── servers/                       # FastAPI Backend
│   ├── app/
│   │   ├── core/
│   │   │   └── audio_processor.py # Giải mã audio, VAD trim, RMS, Chunking
│   │   ├── engine/
│   │   │   └── onnx_engine.py     # Nạp ONNX Runtime (CUDA / CPU fallback)
│   │   ├── schemas.py             # Schema Pydantic gọn gàng
│   │   └── main.py                # FastAPI endpoints & CORS
│   ├── scripts/
│   │   └── export_onnx.py         # Script xuất model_fold_5.pt -> ser_model.onnx
│   └── requirements.txt
│
├── MODELs/                        # Dữ liệu gốc
├── demo-emotion.py                # Demo cũ
└── model_fold_5.pt                # Checkpoint chính
```
