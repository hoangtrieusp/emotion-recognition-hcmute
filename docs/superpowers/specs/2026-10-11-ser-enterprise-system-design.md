# HỆ THỐNG NHẬN DIỆN CẢM XÚC TIẾNG NÓI TOÀN DIỆN (SER ENTERPRISE V2.0)
## ARCHITECTURE & TECHNICAL SPECIFICATION (L0 - L6)

---

### 1. TỔNG QUAN HỆ THỐNG (SYSTEM OVERVIEW)
- **Tên dự án:** ViSEC Speech Emotion Recognition Enterprise Engine
- **Mục tiêu:** Chuyển đổi mô hình nghiên cứu giữa kỳ (Wav2Vec2 + BiGRU + SAP + ArcFace Loss, huấn luyện trên tập ViSEC 4 nhãn: `angry`, `happy`, `neutral`, `sad`) thành một hệ thống Microservices chuẩn Production phục vụ báo cáo Đồ án Tốt nghiệp / Cuối kỳ.
- **Mô hình kiến trúc:** 3 Tiers Microservice (Frontend Client UI, API Gateway & Orchestrator, Hardware-Accelerated AI Engine) tích hợp Model Context Protocol (MCP) cho phép các tác tử Autonomous AI Agent gọi trực tiếp.

---

### 2. PHÂN TÍCH VÀ HÓA GIẢI 3 BẪY KIẾN TRÚC KỸ THUẬT (CRITICAL TRAPS & MITIGATIONS)

#### 2.1. Bẫy L0: ONNX Runtime & BiGRU Dynamic Axes
- **Rủi ro:** Mô hình `SERModel` chứa `BiGRU(input_size=768, hidden_size=384, bidirectional=True)` và `Self-Attention Pooling` trên chuỗi âm thanh có độ dài biến thiên $T$. Khi export sang ONNX, nếu định nghĩa `dynamic_axes` không chặt, ONNX graph sẽ cố định sequence length hoặc bị crash khi suy luận audio có độ dài khác nhau.
- **Giải pháp:**
  - Định nghĩa chặt chẽ `dynamic_axes`:
    ```python
    dynamic_axes = {
        'input_values': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    }
    ```
  - Chạy `torch.onnx.export` với `opset_version=17`.
  - Tích hợp Unit Test tự động kiểm tra sai số số học: So sánh outputs của PyTorch gốc và ONNX Runtime trên tín hiệu ngẫu nhiên, yêu cầu $MAE < 10^{-4}$.
  - Thiết kế kiến trúc **Resilient Dual-Engine**: Ưu tiên chạy `ONNXRuntime` (CUDA -> CPU fallback). Nếu phát hiện file `.onnx` chưa có hoặc có lỗi operator, backend tự động fallback sang `PyTorchEngine` (FP16/FP32) mà không làm gián đoạn hệ thống.

#### 2.2. Bẫy L1/L4: Librosa Bloat & Cold Start Latency
- **Rủi ro:** `librosa` kéo theo hàng loạt dependencies nặng (`scipy`, `audioread`, `numba` compile trên luồng chính). Nếu FastAPI import `librosa`, thời gian Cold Start tăng thêm 3-5 giây và ngốn thêm hàng trăm MB RAM.
- **Giải pháp:**
  - **Tuyệt đối không import `librosa` trong runtime của FastAPI backend.**
  - Sử dụng `soundfile` hoặc `torchaudio` để giải mã trực tiếp dữ liệu âm thanh thành mảng NumPy Float32.
  - Viết thuật toán Voice Activity Detection (VAD) / Silence Trimming và RMS Energy Filter bằng **vectorized NumPy** và **Numba `@njit(fastmath=True)`**.
  - Tốc độ xử lý DSP tăng 10-20x, Cold Start của FastAPI chỉ mất < 1 giây.

#### 2.3. Bẫy L5: Định dạng Thu âm Browser (WebM/Opus vs 16kHz PCM WAV)
- **Rủi ro:** Trình duyệt Chrome/Edge/Safari khi thu âm bằng `MediaRecorder` API xuất ra định dạng `audio/webm` hoặc `audio/ogg` mã hóa bằng Opus Codec, không phải WAV PCM 16kHz mà model yêu cầu.
- **Giải pháp:**
  - Tại API Gateway (FastAPI), tích hợp bộ chuyển đổi đa định dạng sử dụng `pydub` / `ffmpeg` hoặc `soundfile` + `torchaudio.io.StreamReader` / `io.BytesIO`.
  - Chuẩn hóa mọi luồng dữ liệu (WebM, Ogg, MP3, M4A, FLAC, WAV) về dạng chuẩn **16000Hz, Mono, Float32 trong khoảng [-1.0, 1.0]**.

---

### 3. BẢN VẼ KIẾN TRÚC HỆ THỐNG (ASCII ARCHITECTURE BLUEPRINT)

```
===================================================================================
[L5 - CLIENT LAYER] : NEXT.JS 14+ (React, Tailwind CSS, TypeScript, Web Audio API)
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Audio Recorder Component: Thu âm trực tiếp qua mic (Web Audio API)          │
 │ + Real-time Visualizer: Hiển thị waveform dao động âm thanh sống động          │
 │ + Audio Dropzone: Hỗ trợ upload WAV, MP3, M4A, OGG, WEBM                 │
 │ + Emotion Dashboard:                                                          │
 │   - Dominant Emotion Card (Angry 😡, Happy 😄, Neutral 😐, Sad 😢)           │
 │   - Confidence Probability Bars (Hiển thị % phân phối 4 nhãn)                 │
 │   - Acoustic Feature Telemetry (Duration, Sample Rate, RMS Energy)            │
 │   - Latency Monitor (Pre-processing ms, Inference ms, Total E2E ms)           │
 │ + In-Memory Session History: Nghe lại và so sánh kết quả các lần test         │
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ HTTP POST /multipart (Audio Blob)
                                        ▼
===================================================================================
[L4 - GATEWAY & ORCHESTRATION] : FASTAPI (Async ASGI Server - Uvicorn)
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Asynchronous Worker Pool (Không nghẽn I/O khi nhiều user gọi đồng thời)    │
 │ + Universal Audio Ingestor: Tự động giải mã WebM/Opus, MP3, WAV sang PCM      │
 │ + Security & Validation Gate: Kiểm tra payload size (< 25MB), audio headers   │
 │ + Endpoints:                                                                  │
 │   - `GET  /health`                : Trả về trạng thái GPU/CPU, VRAM, RAM       │
 │   - `POST /api/v1/predict/emotion`: Endpoint chính nhận diện cảm xúc          │
 │   - `POST /api/v1/mcp/tools`      : Chuẩn Model Context Protocol cho AI Agent │
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ Raw Float32 Array [1, N]
                                        ▼
===================================================================================
[L1 & L2 - DSP PIPELINE] : NUMBA JIT & VECTORIZED NUMPY
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Vectorized RMS Energy Calculator (JIT compiled)                             │
 │ + Silence Trimmer: Cắt khoảng lặng 2 đầu bằng ngưỡng top_db=30                │
 │ + Quality Gate Filter:                                                        │
 │   - Duration < 0.5s -> Reject (File quá ngắn)                                 │
 │   - RMS < 0.002     -> Reject (Im lặng hoặc chỉ có tiếng ồn)                  │
 │ + Strategic Chunking & Ensemble:                                              │
 │   - Duration <= 3.5s : Inference 1 lần                                        │
 │   - Duration > 3.5s  : Cắt 2 chunk đầu/cuối 3.5s -> Inference -> Mean Pooling │
 └──────────────────────────────────────┬────────────────────────────────────────┘
                                        │ Normalized Tensor (Zero-mean, Unit-var)
                                        ▼
===================================================================================
[L0 - AI INFERENCE ENGINE] : ONNX RUNTIME (Hybrid CUDA / CPU) + PyTorch Fallback
===================================================================================
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │ + Singleton Inference Session: Khởi tạo 1 lần duy nhất lúc khởi động app     │
 │ + Provider Chain: `['CUDAExecutionProvider', 'CPUExecutionProvider']`         │
 │ + Model Graph: Wav2Vec2-Base + BiGRU + SAP + ArcFace Cosine Head              │
 │ + Dynamic Batch & Sequence Length Support                                     │
 │ + Softmax Normalization -> Xác suất 4 cảm xúc                                │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

### 4. CẤU TRÚC MÃ NGUỒN CHI TIẾT (DIRECTORY STRUCTURE)

```
emotion-recognition-hcmute/
├── clients/                       # Next.js 14+ Frontend Client
│   ├── public/
│   │   └── samples/               # 4 file audio mẫu (vui, buồn, tức giận, bình thường)
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx         # Root Layout, Inter Font, Dark/Light Mode Theme
│   │   │   ├── page.tsx           # Dashboard chính điều phối toàn bộ components
│   │   │   └── globals.css        # Tailwind CSS & Keyframes Animation
│   │   ├── components/
│   │   │   ├── Header.tsx         # Navbar với chỉ số Health Status (Online, CUDA/CPU)
│   │   │   ├── AudioRecorder.tsx  # Bộ thu âm mic + Canvas Waveform Visualizer
│   │   │   ├── AudioDropzone.tsx  # Kéo thả file âm thanh + Audio Player preview
│   │   │   ├── EmotionCard.tsx    # Card hiển thị cảm xúc thống trị nổi bật
│   │   │   ├── ProbabilityBars.tsx# Biểu đồ thanh xác suất 4 cảm xúc
│   │   │   ├── TelemetryPanel.tsx # Chỉ số RMS, Duration, Latency (ms)
│   │   │   └── SessionHistory.tsx # Lịch sử các lần test trong phiên làm việc
│   │   ├── hooks/
│   │   │   ├── useAudioRecorder.ts# Quản lý Web Audio API & MediaRecorder
│   │   │   └── useEmotionPredict.ts# Quản lý request POST sang FastAPI Backend
│   │   └── types/
│   │       └── index.ts           # TypeScript interfaces (EmotionResult, Metadata)
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
│
├── servers/                       # FastAPI Backend Service
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       ├── endpoints.py   # Router: /predict/emotion & /health
│   │   │       └── mcp.py         # MCP Tool router cho Autonomous Agents
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings (Paths, Thresholds, Host/Port)
│   │   │   └── audio_processor.py # Numba/NumPy VAD, Trimming, RMS, Normalization
│   │   ├── engine/
│   │   │   ├── base.py            # Abstract Base Inference Engine
│   │   │   ├── onnx_engine.py     # ONNX Runtime Singleton Engine
│   │   │   └── pytorch_engine.py  # PyTorch Backup Fallback Engine
│   │   ├── schemas/
│   │   │   └── emotion.py         # Pydantic Request/Response DTOs
│   │   └── main.py                # FastAPI Application, Lifespan Event, CORS
│   ├── scripts/
│   │   └── export_onnx.py         # Script đóng gói model_fold_5.pt -> ser_model.onnx
│   └── requirements.txt           # fastapi, uvicorn, onnxruntime, soundfile, pydub, numba
│
├── docs/
│   └── superpowers/specs/
│       └── 2026-10-11-ser-enterprise-system-design.md # File đặc tả này
├── MODELs/                        # Dữ liệu huấn luyện & logs gốc (Được bảo toàn)
├── model_fold_5.pt                # Checkpoint chính thức đã huấn luyện
└── demo-emotion.py                # File demo giữa kỳ (Giữ nguyên làm đối chứng)
```

---

### 5. KẾ HOẠCH TRIỂN KHAI THEO GIAI ĐOẠN (IMPLEMENTATION PHASES)

- **Phase 1: Xuất Model ONNX & Kiểm thử Sai số số học (L0)**
  - Viết script `servers/scripts/export_onnx.py`.
  - Xuất `model_fold_5.pt` thành `servers/weights/ser_model.onnx`.
  - Viết tự động parity test so sánh logits PyTorch vs ONNX Runtime ($MAE < 10^{-4}$).
- **Phase 2: Xây dựng Lõi DSP & Backend FastAPI (L1, L2, L4)**
  - Cài đặt `requirements.txt` cho backend.
  - Viết `audio_processor.py` (VAD, silence trim, RMS energy bằng Numba/NumPy, giải mã audio qua soundfile/pydub không dùng librosa).
  - Viết `onnx_engine.py` (Hybrid CUDA/CPU) và `pytorch_engine.py` (Fallback).
  - Hoàn thiện FastAPI routes: `/health`, `/api/v1/predict/emotion`, `/api/v1/mcp/tools`.
  - Chạy backend test với các file audio mẫu.
- **Phase 3: Xây dựng Giao diện Next.js Client (L5)**
  - Khởi tạo Next.js 14+ trong thư mục `clients/`.
  - Xây dựng Web Audio API Visualizer, Audio Recorder, Audio Dropzone.
  - Xây dựng Emotion Dashboard, Confidence Bars, Telemetry Metrics (Latency, RMS).
  - Tạo 4 file mẫu audio test trực tiếp trên UI.
- **Phase 4: Tích hợp Toàn diện & Kiểm thử Tận cùng (E2E Verification)**
  - Chạy đồng thời Frontend & Backend.
  - Kiểm thử tải (Concurrency test) và đo đạc độ trễ E2E.
  - Đảm bảo 100% kịch bản demo chạy trơn tru không lỗi.
