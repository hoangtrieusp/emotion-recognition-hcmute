"use client";

import React, { useState } from "react";
import Header from "../components/Header";
import AudioInput from "../components/AudioInput";
import EmotionDashboard, { PredictionResult } from "../components/EmotionDashboard";
import { Info } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleAnalyze = async (fileOrBlob: File | Blob) => {
    setIsLoading(true);
    setResult(null);

    try {
      const formData = new FormData();
      if (fileOrBlob instanceof File) {
        formData.append("file", fileOrBlob, fileOrBlob.name);
      } else {
        formData.append("file", fileOrBlob, "microphone_recording.webm");
      }

      const response = await fetch("http://127.0.0.1:8000/api/v1/predict/emotion", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        setResult({
          status: "error",
          error: `Lỗi máy chủ (${response.status}): ${errorText}`,
        });
        return;
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setResult({
        status: "error",
        error: `Không thể kết nối đến Backend API (http://127.0.0.1:8000). Vui lòng đảm bảo FastAPI backend đang chạy: ${String(err)}`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Intro banner */}
        <div className="mb-8 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Hệ Thống Đánh Giá & Phân Loại Cảm Xúc Giọng Nói (ViSEC Dataset)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Áp dụng kiến trúc <strong>Wav2Vec2 + BiGRU + Self-Attention Pooling + ArcFace Loss</strong>, tối ưu suy luận bằng <strong>ONNX Runtime</strong> và tiền xử lý <strong>Adaptive Overlapping Chunking</strong>.
            </p>
          </div>
          <div className="flex items-center space-x-2 bg-indigo-50 border border-indigo-100 text-indigo-800 text-xs px-3.5 py-2 rounded-xl shrink-0 font-medium">
            <Info className="w-4 h-4 text-indigo-600 shrink-0" />
            <span>4 Nhãn: Tức giận, Vui vẻ, Bình thường, Buồn bã</span>
          </div>
        </div>

        {/* 2-Column Responsive Dashboard Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: Audio Input & Recording (7 cols) */}
          <div className="lg:col-span-7">
            <AudioInput onAnalyze={handleAnalyze} isLoading={isLoading} />
          </div>

          {/* Right Column: Emotion Analysis Results (5 cols) */}
          <div className="lg:col-span-5">
            <EmotionDashboard result={result} />
          </div>
        </div>
      </main>

      {/* Clean Academic Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500 font-medium">
          Đồ Án Tốt Nghiệp / Báo Cáo Cuối Kỳ · Trường Đại học Sư phạm Kỹ thuật TP. Hồ Chí Minh (HCMUTE) © 2026
        </div>
      </footer>
    </div>
  );
}
