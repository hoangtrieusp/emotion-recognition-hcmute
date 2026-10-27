"use client";

import React from "react";
import { AlertTriangle, Clock, Zap, BarChart2, Flame, Smile, Meh, Frown, RefreshCw } from "lucide-react";

export interface PredictionResult {
  status: "success" | "warning" | "error";
  dominant_emotion?: string | null;
  dominant_probability?: number | null;
  probabilities?: Record<string, number> | null;
  telemetry?: {
    duration: number;
    rms: number;
    num_chunks: number;
    dsp_latency_ms: number;
    inference_latency_ms: number;
    total_latency_ms: number;
  } | null;
  error?: string | null;
}

interface EmotionDashboardProps {
  result: PredictionResult | null;
  isAnalyzing?: boolean;
}

const EMOTION_CONFIG: Record<
  string,
  {
    vietnamese: string;
    english: string;
    icon: React.ReactNode;
    color: string;
    bgColor: string;
    borderColor: string;
    barColor: string;
  }
> = {
  angry: {
    vietnamese: "Tức giận",
    english: "Angry",
    icon: <Flame className="w-8 h-8 text-rose-600" />,
    color: "text-rose-700",
    bgColor: "bg-rose-50",
    borderColor: "border-rose-200",
    barColor: "bg-rose-500",
  },
  happy: {
    vietnamese: "Vui vẻ",
    english: "Happy",
    icon: <Smile className="w-8 h-8 text-emerald-600" />,
    color: "text-emerald-700",
    bgColor: "bg-emerald-50",
    borderColor: "border-emerald-200",
    barColor: "bg-emerald-500",
  },
  neutral: {
    vietnamese: "Bình thường",
    english: "Neutral",
    icon: <Meh className="w-8 h-8 text-slate-600" />,
    color: "text-slate-700",
    bgColor: "bg-slate-100",
    borderColor: "border-slate-200",
    barColor: "bg-slate-500",
  },
  sad: {
    vietnamese: "Buồn bã",
    english: "Sad",
    icon: <Frown className="w-8 h-8 text-indigo-600" />,
    color: "text-indigo-700",
    bgColor: "bg-indigo-50",
    borderColor: "border-indigo-200",
    barColor: "bg-indigo-500",
  },
};

export default function EmotionDashboard({ result, isAnalyzing }: EmotionDashboardProps) {
  if (isAnalyzing) {
    return (
      <div className="bg-white rounded-2xl border border-indigo-200 p-8 shadow-sm flex flex-col items-center justify-center text-center h-full min-h-[380px] space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center animate-pulse">
          <RefreshCw className="w-8 h-8 animate-spin" />
        </div>
        <div>
          <h3 className="text-base font-bold text-slate-800">
            Đang Phân Tích Cảm Xúc Giọng Nói...
          </h3>
          <p className="text-xs text-slate-500 max-w-sm mt-1">
            Hệ thống đang tiếp nhận và phân tích dữ liệu âm thanh giọng nói.
          </p>
        </div>
        <div className="w-full max-w-xs space-y-2 pt-2">
          <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
            <div className="h-full bg-indigo-600 animate-pulse rounded-full w-3/4" />
          </div>
          <span className="text-[11px] text-indigo-600 font-medium">Đang xử lý âm thanh</span>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-8 shadow-sm flex flex-col items-center justify-center text-center h-full min-h-[380px]">
        <div className="w-16 h-16 rounded-2xl bg-indigo-50 text-indigo-500 flex items-center justify-center mb-4">
          <BarChart2 className="w-8 h-8" />
        </div>
        <h3 className="text-base font-bold text-slate-800">
          Chưa Có Dữ Liệu Phân Tích
        </h3>
        <p className="text-xs text-slate-500 max-w-sm mt-1">
          Vui lòng thu âm trực tiếp hoặc tải file âm thanh lên để bắt đầu phân tích cảm xúc.
        </p>
      </div>
    );
  }

  // Error State (e.g. Backend offline, network failure, 500 error)
  if (result.status === "error") {
    return (
      <div className="bg-white rounded-2xl border border-rose-200 p-6 shadow-sm">
        <div className="flex items-start space-x-3 bg-rose-50 border border-rose-200 p-4 rounded-xl text-rose-900 mb-5">
          <AlertTriangle className="w-6 h-6 text-rose-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold text-sm">Lỗi Kết Nối Máy Chủ API</h4>
            <p className="text-xs mt-1 text-rose-800 leading-relaxed">{result.error}</p>
          </div>
        </div>

        <div className="bg-slate-50 border border-slate-200 p-4 rounded-xl text-xs space-y-2">
          <p className="font-semibold text-slate-800">
            👉 Hướng dẫn khắc phục (dành cho lập trình viên / người chạy):
          </p>
          <ol className="list-decimal list-inside space-y-1 text-slate-600 font-mono text-[11px]">
            <li>Mở một Terminal mới tại thư mục gốc của project.</li>
            <li>Kích hoạt môi trường Python (nếu có venv).</li>
            <li>Chạy lệnh: <span className="text-indigo-600 font-bold bg-indigo-50 px-1.5 py-0.5 rounded">python -m uvicorn servers.app.main:app --port 8000</span></li>
            <li>Đảm bảo file trọng số <span className="text-amber-700 font-bold">servers/weights/ser_model.onnx</span> đã tồn tại.</li>
          </ol>
        </div>
      </div>
    );
  }

  // Warning (File too short or low RMS)
  if (result.status === "warning") {
    return (
      <div className="bg-white rounded-2xl border border-amber-200 p-6 shadow-sm">
        <div className="flex items-start space-x-3 bg-amber-50 border border-amber-200 p-4 rounded-xl text-amber-900 mb-6">
          <AlertTriangle className="w-6 h-6 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <h4 className="font-bold text-sm">Cảnh báo Chất lượng Âm thanh</h4>
            <p className="text-xs mt-1 text-amber-800">{result.error}</p>
          </div>
        </div>

        {result.telemetry && (
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
              <span className="text-slate-500">Thời lượng đo được:</span>
              <p className="font-bold text-slate-800 text-sm mt-0.5">{result.telemetry.duration}s</p>
            </div>
            <div className="bg-slate-50 p-3 rounded-xl border border-slate-200">
              <span className="text-slate-500">Năng lượng RMS:</span>
              <p className="font-bold text-slate-800 text-sm mt-0.5">{result.telemetry.rms}</p>
            </div>
          </div>
        )}
      </div>
    );
  }

  const dominant = result.dominant_emotion || "neutral";
  const dominantConfig = EMOTION_CONFIG[dominant] || EMOTION_CONFIG.neutral;
  const dominantPercentage = result.dominant_probability
    ? (result.dominant_probability * 100).toFixed(1)
    : "0.0";

  return (
    <div className="space-y-5">
      {/* Dominant Emotion Hero Card */}
      <div
        className={`rounded-2xl border ${dominantConfig.borderColor} ${dominantConfig.bgColor} p-6 shadow-sm flex items-center justify-between`}
      >
        <div className="flex items-center space-x-4">
          <div className="w-16 h-16 rounded-2xl bg-white shadow-sm flex items-center justify-center border border-slate-100">
            {dominantConfig.icon}
          </div>
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-slate-500">
              Cảm xúc chính
            </span>
            <div className="flex items-baseline space-x-2 mt-0.5">
              <h2 className={`text-2xl font-black ${dominantConfig.color}`}>
                {dominantConfig.vietnamese}
              </h2>
              <span className="text-sm font-semibold text-slate-500">
                ({dominantConfig.english})
              </span>
            </div>
          </div>
        </div>

        <div className="text-right">
          <span className="text-xs font-semibold text-slate-500">Độ tin cậy</span>
          <p className={`text-3xl font-black ${dominantConfig.color}`}>
            {dominantPercentage}%
          </p>
        </div>
      </div>

      {/* Probabilities Distribution Bars */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
          Phân bố xác suất 4 cảm xúc:
        </h4>

        <div className="space-y-3.5">
          {["angry", "happy", "neutral", "sad"].map((emoKey) => {
            const config = EMOTION_CONFIG[emoKey];
            const prob = result.probabilities ? result.probabilities[emoKey] || 0 : 0;
            const percentage = (prob * 100).toFixed(1);

            return (
              <div key={emoKey} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center space-x-1.5 font-semibold text-slate-700">
                    <span>{config.vietnamese} ({config.english})</span>
                  </div>
                  <span className="font-bold text-slate-900">{percentage}%</span>
                </div>
                <div className="w-full bg-slate-100 h-2.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${config.barColor} transition-all duration-500 rounded-full`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Technical Telemetry & Latency Card */}
      {result.telemetry && (
        <div className="bg-white rounded-2xl border border-slate-200 p-5 shadow-sm">
          <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-3 flex items-center space-x-1.5">
            <Zap className="w-3.5 h-3.5 text-indigo-600" />
            <span>Thông số đo lường hiệu năng (Telemetry & Latency):</span>
          </h4>

          <div className="grid grid-cols-3 gap-2.5 text-center">
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <span className="text-[11px] text-slate-500 block font-medium">Thời lượng Audio</span>
              <span className="font-bold text-xs text-slate-800">
                {result.telemetry.duration}s
              </span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <span className="text-[11px] text-slate-500 block font-medium">Năng lượng RMS</span>
              <span className="font-bold text-xs text-slate-800">
                {result.telemetry.rms}
              </span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <span className="text-[11px] text-slate-500 block font-medium">Số Chunks trượt</span>
              <span className="font-bold text-xs text-slate-800">
                {result.telemetry.num_chunks}
              </span>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-slate-100 grid grid-cols-3 gap-2.5 text-center">
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <span className="text-[11px] text-slate-500 block font-medium">DSP Latency</span>
              <span className="font-bold text-xs text-slate-800">
                {result.telemetry.dsp_latency_ms} ms
              </span>
            </div>
            <div className="bg-slate-50 p-2.5 rounded-xl border border-slate-100">
              <span className="text-[11px] text-slate-500 block font-medium">ONNX Latency</span>
              <span className="font-bold text-xs text-indigo-700">
                {result.telemetry.inference_latency_ms} ms
              </span>
            </div>
            <div className="bg-emerald-50 p-2.5 rounded-xl border border-emerald-200">
              <span className="text-[11px] text-emerald-700 block font-semibold">Total E2E</span>
              <span className="font-extrabold text-xs text-emerald-800">
                {result.telemetry.total_latency_ms} ms
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
