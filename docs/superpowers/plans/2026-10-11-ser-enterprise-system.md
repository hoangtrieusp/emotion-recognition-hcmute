# ViSEC SER Enterprise v2.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống Nhận diện Cảm xúc Tiếng nói Microservices hoàn chỉnh gồm Next.js Frontend UI (Web Audio visualizer, Emotion Dashboard) và FastAPI ASGI Backend (ONNX Runtime GPU/CPU, Numba/NumPy VAD & Adaptive Overlapping Chunking, MCP Tool router) thay thế demo Gradio nguyên khối.

**Architecture:** Tách biệt 3 tầng (L5 Client UI Next.js, L4 API Gateway FastAPI, L1-L2 DSP & L0 AI Engine ONNX). Kết hợp Resilient Dual-Engine (ONNX CUDA -> CPU fallback -> PyTorch fallback) và Universal Audio Ingestor (giải mã WebM/Opus, MP3, WAV sang 16kHz Float32).

**Tech Stack:** Next.js 14+ (TypeScript, Tailwind CSS, Web Audio API), FastAPI, Uvicorn, ONNX Runtime (`onnxruntime-gpu` / `onnxruntime`), PyTorch 2.5, Numba, NumPy, SoundFile, PyDub / FFmpeg, Model Context Protocol (MCP).

**Spec:** `docs/superpowers/specs/2026-10-11-ser-enterprise-system-design.md`

## Global Constraints
- Target Audio Format: 16000Hz, Mono, Float32 in range [-1.0, 1.0].
- Emotion Classes: 4 classes strictly mapped as `{0: "angry", 1: "happy", 2: "neutral", 3: "sad"}`.
- Checkpoint source: `model_fold_5.pt` containing `model_state` and `criterion_state`.
- Parity Tolerance: Max Absolute Error between PyTorch FP32 and ONNX Runtime must be $< 10^{-4}$.
- Zero Librosa on FastAPI Runtime: Cấm import `librosa` trong runtime của Backend API để triệt tiêu Cold Start lag.

## Review Focus
1. Non-WAV inputs from browser MediaRecorder (WebM/Opus or Ogg) decoded cleanly without data corruption.
2. Silent or low-energy audio (RMS $< 0.002$) rejected gracefully with 400 Bad Request and descriptive message.
3. Audio $> 3.5$s segmented into overlapping windows (3.5s window, 2.0s hop) without temporal blind spots.
4. Sudden emotional spike preserved via Confidence-Weighted Salience Pooling instead of arithmetic dilution.
5. Systems lacking NVIDIA CUDA automatically switch to CPUExecutionProvider with zero downtime or crash.

---

### Task 1: AI Model ONNX Exporter & Parity Validation (L0)

**Files:**
- Create: `servers/scripts/export_onnx.py`
- Create: `servers/tests/test_onnx_parity.py`
- Target Artifact: `servers/weights/ser_model.onnx`

**Interfaces:**
- Consumes: `model_fold_5.pt` checkpoint with keys `model_state` and `criterion_state`.
- Produces: `ser_model.onnx` with input `input_values: [batch_size, sequence_length]` and output `logits: [batch_size, 4]`.

- [ ] **Step 1: Write test to verify ONNX parity against PyTorch**
- [ ] **Step 2: Run test to verify it fails before export**
- [ ] **Step 3: Implement `export_onnx.py` with EndToEndSERModel combining Wav2Vec2, BiGRU, SAP and ArcFace Head**
- [ ] **Step 4: Execute `export_onnx.py` and run parity validation test ($MAE < 10^{-4}$)**
- [ ] **Step 5: Commit exported script and tests**

---

### Task 2: DSP Audio Engine & Numba/NumPy Preprocessing (L1, L2)

**Files:**
- Create: `servers/app/core/audio_processor.py`
- Create: `servers/tests/test_audio_processor.py`

**Interfaces:**
- Consumes: Raw audio bytes (WAV, MP3, WebM, OGG).
- Produces: `AudioProcessingResult(chunks: np.ndarray, duration: float, sample_rate: int, rms_energy: float, is_valid: bool, error_message: Optional[str])`.

- [ ] **Step 1: Write unit tests for audio decoding, trimming, overlapping chunking, and salience pooling**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `audio_processor.py` using soundfile/pydub, Numba JIT RMS, and Adaptive Overlapping Chunking**
- [ ] **Step 4: Run unit tests to verify 100% pass across sample audio arrays and corrupt files**
- [ ] **Step 5: Commit DSP audio processor**

---

### Task 3: Resilient Dual-Engine Inference Service (L0, L4)

**Files:**
- Create: `servers/app/engine/base.py`
- Create: `servers/app/engine/onnx_engine.py`
- Create: `servers/app/engine/pytorch_engine.py`
- Create: `servers/app/engine/factory.py`
- Create: `servers/tests/test_engine.py`

**Interfaces:**
- Consumes: Preprocessed audio chunks `(K, sequence_length)`.
- Produces: Emotion prediction dictionary `Dict[str, float]` with emotion probabilities summing to 1.0.

- [ ] **Step 1: Write unit tests for engine initialization, fallback trigger, and warm-up inference**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement `ONNXEngine` (Hybrid CUDA/CPU) and `PyTorchEngine` (fallback)**
- [ ] **Step 4: Run tests to verify seamless fallback and exact prediction output**
- [ ] **Step 5: Commit inference engine module**

---

### Task 4: FastAPI Gateway, Schemas & MCP Tool Endpoint (L4, L6)

**Files:**
- Create: `servers/app/schemas/emotion.py`
- Create: `servers/app/core/config.py`
- Create: `servers/app/api/v1/endpoints.py`
- Create: `servers/app/api/v1/mcp.py`
- Create: `servers/app/main.py`
- Create: `servers/requirements.txt`
- Create: `servers/tests/test_api.py`

**Interfaces:**
- Consumes: HTTP Multipart audio file upload.
- Produces: JSON response with `status`, `dominant_emotion`, `probabilities`, `telemetry` (latency_ms, rms, duration), and MCP tool schema.

- [ ] **Step 1: Write integration tests for `/health`, `/api/v1/predict/emotion`, and `/api/v1/mcp/tools`**
- [ ] **Step 2: Run test to verify failure**
- [ ] **Step 3: Implement schemas, endpoints, lifespan handler, CORS middleware, and main application**
- [ ] **Step 4: Run integration tests with dummy audio file to verify successful prediction response**
- [ ] **Step 5: Commit FastAPI backend module**

---

### Task 5: Next.js Frontend Foundation & Web Audio Visualizer (L5)

**Files:**
- Create: `clients/package.json`
- Create: `clients/src/app/layout.tsx`
- Create: `clients/src/app/globals.css`
- Create: `clients/src/components/Header.tsx`
- Create: `clients/src/components/AudioRecorder.tsx`
- Create: `clients/src/components/WaveformVisualizer.tsx`
- Create: `clients/src/hooks/useAudioRecorder.ts`

**Interfaces:**
- Consumes: Browser MediaStream microphone input.
- Produces: Audio Blob (WebM/WAV) and real-time live canvas waveform visualizer.

- [ ] **Step 1: Scaffold Next.js client with TypeScript, Tailwind CSS, Lucide icons**
- [ ] **Step 2: Implement `useAudioRecorder` hook managing microphone state, duration timer, and blob export**
- [ ] **Step 3: Implement real-time canvas `WaveformVisualizer` reacting to microphone frequency data**
- [ ] **Step 4: Implement `Header` displaying API health status and hardware acceleration mode**
- [ ] **Step 5: Verify build with `npm run build` or `npm run lint`**
- [ ] **Step 6: Commit frontend audio recorder module**

---

### Task 6: Audio Dropzone, Preset Samples & Emotion Dashboard (L5)

**Files:**
- Create: `clients/src/components/AudioDropzone.tsx`
- Create: `clients/src/components/EmotionCard.tsx`
- Create: `clients/src/components/ProbabilityBars.tsx`
- Create: `clients/src/components/TelemetryPanel.tsx`
- Create: `clients/src/components/SessionHistory.tsx`
- Create: `clients/src/app/page.tsx`
- Create: `clients/public/samples/` (4 sample audio clips)

**Interfaces:**
- Consumes: Backend API response `EmotionPredictionResponse`.
- Produces: Interactive Dashboard with dynamic theme, probability bars, acoustic telemetry, and in-memory test history.

- [ ] **Step 1: Implement `AudioDropzone` with drag-and-drop and audio preview playback**
- [ ] **Step 2: Implement `EmotionCard` with color themes: Angry (Red), Happy (Amber), Neutral (Slate), Sad (Blue)**
- [ ] **Step 3: Implement `ProbabilityBars` and `TelemetryPanel` showing Latency breakdown (Pre-processing ms, Inference ms)**
- [ ] **Step 4: Implement `SessionHistory` for in-memory review and comparison**
- [ ] **Step 5: Assemble `page.tsx` linking Recorder, Dropzone, and Dashboard**
- [ ] **Step 6: Verify frontend layout and build**
- [ ] **Step 7: Commit complete frontend application**

---

### Task 7: End-to-End System Verification & Benchmark (L7)

**Files:**
- Create: `servers/tests/test_e2e.py`

**Interfaces:**
- Consumes: Running FastAPI backend and test audio files.
- Produces: Benchmark report measuring E2E latency, throughput, and correctness across all 4 emotions.

- [ ] **Step 1: Launch FastAPI backend server**
- [ ] **Step 2: Execute E2E benchmark script testing WAV, MP3, WebM audio formats**
- [ ] **Step 3: Verify latency metrics (Pre-processing $< 50$ms, ONNX inference $< 150$ms on GPU / $< 350$ms on CPU)**
- [ ] **Step 4: Document startup and demo instructions in `README.md`**
- [ ] **Step 5: Final commit and audit**
