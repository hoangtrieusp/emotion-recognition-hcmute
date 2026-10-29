"use client";

import React, { useState } from "react";
import Header from "../components/Header";
import AudioInput from "../components/AudioInput";
import EmotionDashboard, { PredictionResult } from "../components/EmotionDashboard";
import { Info } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const abortControllerRef = React.useRef<AbortController | null>(null);

  const handleAnalyze = async (fileOrBlob: File | Blob) => {
    if (isAnalyzing) return;

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsAnalyzing(true);
    setResult(null);

    try {
      const formData = new FormData();
      if (fileOrBlob instanceof File) {
        formData.append("file", fileOrBlob, fileOrBlob.name);
      } else {
        const rawType = fileOrBlob.type || "audio/webm";
        const cleanType = rawType.includes("wav") ? "audio/wav" : "audio/webm";
        const extension = cleanType === "audio/wav" ? "wav" : "webm";
        const audioFile = new File([fileOrBlob], `microphone_recording.${extension}`, {
          type: cleanType,
        });
        formData.append("file", audioFile, audioFile.name);
      }

      const response = await fetch("http://127.0.0.1:8000/api/v1/predict/emotion", {
        method: "POST",
        body: formData,
        signal: controller.signal,
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
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") {
        return;
      }
      setResult({
        status: "error",
        error: `Không thể kết nối đến Backend API (http://127.0.0.1:8000). Vui lòng đảm bảo FastAPI backend đang chạy: ${String(err)}`,
      });
    } finally {
      setIsAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50">
      <Header />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8 bg-white border border-slate-200 rounded-2xl p-5 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              Nhận diện cảm xúc giọng nói tiếng Việt
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Hệ thống hỗ trợ phân tích và nhận diện 4 trạng thái cảm xúc: Tức giận, Vui vẻ, Bình thường và Buồn bã.
            </p>
          </div>
          <div className="flex items-center space-x-2 bg-indigo-50 border border-indigo-100 text-indigo-800 text-xs px-3.5 py-2 rounded-xl shrink-0 font-medium">
            <Info className="w-4 h-4 text-indigo-600 shrink-0" />
            <span>4 Nhãn: Tức giận, Vui vẻ, Bình thường, Buồn bã</span>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          <div className="lg:col-span-7">
            <AudioInput onAnalyze={handleAnalyze} isAnalyzing={isAnalyzing} />
          </div>

          <div className="lg:col-span-5">
            <EmotionDashboard result={result} isAnalyzing={isAnalyzing} />
          </div>
        </div>
      </main>

      <footer className="bg-white border-t border-slate-200 py-4 mt-auto">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500 font-medium">
          Hệ Thống Nhận Diện Cảm Xúc Giọng Nói Tiếng Việt © 2026
        </div>
      </footer>
    </div>
  );
}
