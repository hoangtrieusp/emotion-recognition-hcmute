import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Hệ Thống Nhận Diện Cảm Xúc",
  description: "Hệ thống nhận diện cảm xúc qua giọng nói tiếng Việt",
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
