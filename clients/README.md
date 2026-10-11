# Giao Diện Nhận Diện Cảm Xúc Giọng Nói Tiếng Việt (Next.js)

> **Hệ Thống Nhận Diện Cảm Xúc Qua Giọng Nói Tiếng Việt**

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
│   ├── layout.tsx                # Khung giao diện chuẩn, tối ưu font & metadata
│   ├── page.tsx                  # Controller chính, quản lý state isAnalyzing & gọi API
│   └── globals.css               # Styling tokens, thanh cuộn tùy biến
├── components/
│   ├── Header.tsx                # Thanh tiêu đề và trạng thái kết nối
│   ├── WaveformVisualizer.tsx    # Dao động ký HTML5 Canvas khi thu âm Micro
│   ├── AudioInput.tsx            # Thu âm Micro & Tải lên file âm thanh
│   └── EmotionDashboard.tsx      # Thẻ Cảm xúc chính, 4 Thanh Xác suất & Thời gian xử lý
└── hooks/
    └── useAudioRecorder.ts       # Hook Web Audio API (Chuẩn hóa Blob 'audio/webm'/'audio/wav')
```

---

## 3. Các Tính Năng Nổi Bật

1. **Khóa Chống Race Condition (`isAnalyzing` Lock):**
   - Tự động vô hiệu hóa toàn bộ tương tác (nút Thu âm, Tải file) khi đang phân tích để tránh xung đột dữ liệu hoặc giao diện nhảy loạn xạ khi click dồn dập.
   - Tích hợp `AbortController` để hủy bỏ request cũ nếu có yêu cầu mới.

2. **Hook Thu âm Chuẩn Hóa (`useAudioRecorder`):**
   - Tự động phát hiện codec trình duyệt hỗ trợ (`audio/webm;codecs=opus`, `audio/webm`, `audio/wav`).
   - Đóng gói file `Blob` với MIME type chuẩn xác trước khi nạp vào `FormData`.
   - Tự động giải phóng `MediaStreamTrack` ngay khi dừng thu âm để trình duyệt tắt icon Micro.

3. **Thông Số Xử Lý Thời Gian Thực:**
   - Hiển thị phân tách chi tiết: Thời lượng âm thanh đo được, Năng lượng RMS, Thời gian tiền xử lý (ms), Thời gian chạy mô hình AI (ms), và Tổng thời gian xử lý (ms).

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
