# ViSEC SER Frontend Client (Next.js 16 + React 19)

> **Giao diện Người dùng Phân loại Cảm xúc Giọng nói Tiếng Việt (ViSEC Dataset)**
> **Đồ án cuối kỳ trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE)**

---

## 1. Công nghệ & Thư viện (Tech Stack)

- **Framework:** [Next.js 16](https://nextjs.org) (App Router, Turbopack)
- **UI Library:** [React 19](https://react.dev)
- **Styling:** [Tailwind CSS v4](https://tailwindcss.com) (Modern CSS variables, Light Mode)
- **Icons:** [Lucide React](https://lucide.dev)
- **Web Audio API:** Tích hợp `AudioContext` và `AnalyserNode` cho visualizer dao động sóng âm thanh thời gian thực.

---

## 2. Cấu trúc Thành phần (Architecture & Components)

```text
clients/src/
├── app/
│   ├── layout.tsx                # Khung giao diện chuẩn, tối ưu SEO & font
│   ├── page.tsx                  # Controller chính, quản lý state isAnalyzing & gọi API
│   └── globals.css               # Styling tokens, thanh cuộn tùy biến
├── components/
│   ├── Header.tsx                # Thanh tiêu đề + Badge kiểm tra trạng thái Backend/GPU
│   ├── WaveformVisualizer.tsx    # Dao động ký HTML5 Canvas khi thu âm Micro
│   ├── AudioInput.tsx            # Thu âm Micro, Kéo thả file & 4 Nút Test Mẫu Nhanh
│   └── EmotionDashboard.tsx      # Thẻ Cảm xúc Thống trị, 4 Thanh Xác suất & Telemetry
└── hooks/
    └── useAudioRecorder.ts       # Hook Web Audio API (Chuẩn hóa Blob 'audio/webm'/'audio/wav')
```

---

## 3. Các Tính Năng Nổi Bật

1. **Khóa Chống Race Condition (`isAnalyzing` Lock):**
   - Tự động vô hiệu hóa toàn bộ tương tác (nút Test Nhanh, nút Thu âm, Dropzone) khi đang phân tích để tránh xung đột dữ liệu hoặc giao diện nhảy loạn xạ khi click dồn dập.
   - Tích hợp `AbortController` để hủy bỏ request cũ nếu có yêu cầu mới.

2. **Hook Thu âm Chuẩn Hóa (`useAudioRecorder`):**
   - Tự động phát hiện codec trình duyệt hỗ trợ (`audio/webm;codecs=opus`, `audio/webm`, `audio/wav`).
   - Đóng gói file `Blob` với MIME type chuẩn xác trước khi nạp vào `FormData`.
   - Tự động giải phóng `MediaStreamTrack` ngay khi dừng thu âm để trình duyệt tắt icon Micro.

3. **Chỉ số Kỹ thuật Thời gian Thực (Production Telemetry):**
   - Hiển thị phân tách chi tiết: Thời lượng âm thanh đo được, Năng lượng RMS, Độ trễ DSP (ms), Độ trễ suy luận ONNX (ms), và Tổng thời gian E2E (ms).

---

## 4. Hướng dẫn Cài đặt & Khởi chạy

### Cài đặt dependencies (nếu clone mới):
```bash
npm install
```

### Chạy môi trường Phát triển (Development Server):
```bash
npm run dev
```
Truy cập giao diện tại: [http://localhost:3000](http://localhost:3000)

### Biên dịch Production Build (Kiểm tra Type-Safety):
```bash
npm run build
```
Đạt chuẩn **100% Type-safe**, 0 lỗi TypeScript và ESLint.
