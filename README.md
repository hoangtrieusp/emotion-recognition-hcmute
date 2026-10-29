# ViSEC Speech Emotion Recognition (SER) System

Hệ thống nhận diện cảm xúc giọng nói tiếng Việt (Vietnamese Speech Emotion Recognition) hỗ trợ phân loại 4 trạng thái cảm xúc: **Tức giận (Angry)**, **Vui vẻ (Happy)**, **Bình thường (Neutral)** và **Buồn bã (Sad)**.

Hệ thống gồm 3 phần chính:
- **Core AI**: Mô hình Wav2Vec2 + BiGRU + Self-Attention Pooling + ArcFace Head.
- **Backend API**: FastAPI + DSP tiền xử lý lọc nhiễu thực tế (tiếng quạt, tiếng ồn, lọc phi ngôn ngữ) + ONNX Runtime tăng tốc suy luận.
- **Frontend Dashboard**: Web App Next.js 16 (React 19 + Tailwind CSS) hỗ trợ thu âm micro trực tiếp hoặc tải file âm thanh.

---

## Yêu Cầu Môi Trường

- **Python**: 3.10 hoặc 3.11
- **Node.js**: 18.x trở lên
- **FFmpeg**: Đã cài đặt hoặc thông qua thư viện `imageio-ffmpeg`

---

## Hướng Dẫn Cài Đặt & Sử Dụng

### Bước 1: Chuẩn Bị Trọng Số Mô Hình (Weights)

Do các tệp mô hình lớn (`.pt`, `.onnx`) không lưu trực tiếp trên Git, bạn cần chuẩn bị trọng số theo một trong hai cách sau:

#### Cách A: Tự huấn luyện bằng Notebooks trong `MODELs/`
Mở thư mục `MODELs/colabs/` và chạy trên Google Colab / GPU cục bộ:
1. `setup_colab.ipynb`: Cài đặt môi trường và tải tập dữ liệu ViSEC.
2. `data_ preprocessing.ipynb`: Tiền xử lý dữ liệu âm thanh.
3. `training_model.ipynb`: Huấn luyện mô hình Wav2Vec2 + ArcFace theo phương pháp 5-Fold Cross Validation. Sau khi huấn luyện, bạn sẽ thu được các file checkpoint (ví dụ `model_fold_5.pt`).

#### Cách B: Chuyển đổi Checkpoint sang ONNX Runtime (Khuyên dùng)
Khi đã có file checkpoint `model_fold_5.pt` (đặt tại thư mục gốc hoặc `MODELs/model/`):
```bash
# Chuyển đổi mô hình PyTorch sang ONNX tối ưu
python servers/scripts/export_onnx.py
```
Lệnh trên sẽ tự động xuất mô hình đã tối ưu sang `servers/weights/ser_model.onnx`.

---

### Bước 2: Khởi Chạy Backend (FastAPI)

Mở Terminal 1 tại thư mục gốc dự án:
```powershell
# 1. Cài đặt các thư viện phụ thuộc
pip install -r servers/requirements.txt

# 2. Khởi chạy server FastAPI (Port 8000)
python -m uvicorn servers.app.main:app --host 127.0.0.1 --port 8000 --reload
```
- **API Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Kiểm tra trạng thái:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Bước 3: Khởi Chạy Frontend (Next.js)

Mở Terminal 2:
```powershell
# 1. Chuyển vào thư mục clients
cd clients

# 2. Cài đặt node_modules (chỉ chạy lần đầu)
npm install

# 3. Khởi chạy giao diện Next.js (Port 3000)
npm run dev
```
- **Giao diện Web:** Mở trình duyệt tại [http://localhost:3000](http://localhost:3000).
- Bạn có thể nhấn nút thu âm trực tiếp từ microphone hoặc kéo thả file âm thanh (MP3, WAV, WebM, M4A) để phân tích cảm xúc thời gian thực.

---

## Kiểm Thử Hệ Thống (Automated Tests)

Chạy bộ kiểm thử tự động kiểm tra độ bền vững trong môi trường thực tế (lọc tạp âm quạt, tiếng ồn, nhạc nền, từ chối phi ngôn ngữ):
```powershell
python -m pytest servers/tests/test_realworld_robustness.py -v
```
