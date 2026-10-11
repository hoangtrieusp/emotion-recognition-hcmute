import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ViSEC SER Enterprise - Hệ Thống Nhận Diện Cảm Xúc Tiếng Nói",
  description: "Hệ thống nhận diện cảm xúc giọng nói thời gian thực dựa trên Wav2Vec2, BiGRU, SAP và ArcFace Head - HCMUTE Capstone Project",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="vi" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">{children}</body>
    </html>
  );
}
