# ViSEC Speech Emotion Recognition (SER) Enterprise v2.0
> **Đồ Án Cuối Kỳ / Khóa Luận Tốt Nghiệp · Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE)**  
> **Kiến trúc:** Next.js (Light Theme) + FastAPI (ASGI) + ONNX Runtime (CUDA / CPU Fallback) + Adaptive DSP Chunking

---

## 1. Giới thiệu Tổng quan
Hệ thống nhận diện cảm xúc giọng nói tiếng Việt dựa trên tập dữ liệu **ViSEC**, chuyển đổi thành công từ mô hình nghiên cứu giữa kỳ (Gradio đơn luồng) sang kiến trúc **Microservices chuẩn Production**:
- **4 Cảm xúc nhận diện:** 🔥 **Tức giận (Angry)**, 😄 **Vui vẻ (Happy)**, 😐 **Bình thường (Neutral)**, 😢 **Buồn bã (Sad)**.
- **Mô hình AI Lõi:** Wav2Vec2-Base (13-layer Weighted Pooling) $\rightarrow$ BiGRU $\rightarrow$ Self-Attention Pooling (Mean + Std) $\rightarrow$ ArcFace Cosine Head (256-dim embedding).
- **Tối ưu hóa suy luận:** Chuyển đổi toàn bộ mô hình PyTorch thành đồ thị tính toán tĩnh **ONNX** với `dynamic_axes`, hỗ trợ **NVIDIA CUDA GPU** và tự động fallback sang **CPU**.
- **Độ trễ toàn hệ thống:** **~150ms** trên GPU và **~200-300ms** trên CPU, hoàn toàn loại bỏ độ trễ khởi động của thư viện nặng.

---

## 2. Kiến trúc Hệ thống (L0 - L5)

```
[L5: Client UI]        Next.js 14+ (Light Theme, Web Audio API Mic Oscilloscope, 4 Presets)
                             │
                             ▼ HTTP Multipart POST (WAV, MP3, WebM/Opus)
[L4: API Gateway]      FastAPI (Uvicorn ASGI, Non-blocking I/O)
                             │
                             ▼ Float32 NumPy Array (16kHz Mono)
[L1-L2: DSP Engine]    SoundFile/PyDub Decode + Silence Trim (30dB) + Overlapping Window (3.5s, stride 2s)
                             │
                             ▼ Dynamic Batch Array (K, 56000)
[L0: AI Engine]        ONNX Runtime (CUDAExecutionProvider -> CPU Fallback) -> Salience Pooling
```

---

## 3. Hướng dẫn Khởi chạy Hệ thống (Chỉ 2 Bước)

### Bước 1: Khởi động Backend FastAPI
Mở Terminal 1 tại thư mục gốc của dự án:
```bash
# Cài đặt thư viện backend (nếu chưa cài)
pip install -r servers/requirements.txt

# Khởi chạy server FastAPI tại cổng 8000
python -m uvicorn servers.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Docs Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Bước 2: Khởi động Frontend Next.js
Mở Terminal 2 tại thư mục `clients/`:
```bash
cd clients
npm run dev
```
- Mở trình duyệt tại: [http://localhost:3000](http://localhost:3000)

---

## 4. Các Điểm Sáng Kỹ thuật Dành Cho Báo Cáo Hội Đồng

1. **Khắc phục Triệt để Điểm Mù Thời Gian (Temporal Blind Spot):**
   - Thay vì cắt thô 2 đầu của file dài như bản demo giữa kỳ, hệ thống áp dụng **Adaptive Overlapping Sliding Window** (cửa sổ 3.5s, bước nhảy 2.0s), đảm bảo bao phủ 100% thời lượng câu nói.
   - Kết hợp **Confidence-Weighted Salience Pooling** để bảo toàn các cảm xúc bộc phát ngắn hạn (tiếng quát giận, tiếng cười) không bị loãng bởi các đoạn nói bình thường.

2. **Tối ưu Hóa Khởi Động & Tiền Xử Lý (Zero Librosa Runtime):**
   - Loại bỏ việc import `librosa` trên FastAPI backend giúp triệt tiêu hoàn toàn Cold Start lag (thời gian khởi động backend giảm từ 4s xuống dưới 0.8s).
   - Giải mã âm thanh đa định dạng (đặc biệt là WebM/Opus từ trình duyệt) về chuẩn PCM 16kHz Float32.

3. **Khả năng Phục hồi Cao (High Availability):**
   - Cơ chế tự động phát hiện phần cứng: Ưu tiên `CUDAExecutionProvider` nếu có GPU NVIDIA, tự động chuyển về `CPUExecutionProvider` nếu chạy trên laptop trình chiếu không card rời.
   - Kiểm định sai số số học tuyệt đối ($MAE < 10^{-7}$) giữa PyTorch gốc và ONNX Runtime.

4. **Trực Quan Hóa & Khả năng Quan Sát (Observability):**
   - Dashboard hiển thị phân tách chi tiết độ trễ: **DSP Latency (ms)**, **ONNX Inference Latency (ms)** và **Total E2E (ms)**.
   - Giao diện Sáng (Clean Light Theme) với độ tương phản cao, tối ưu hiển thị trên màn chiếu của phòng hội đồng.
