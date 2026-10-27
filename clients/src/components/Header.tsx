"use client";

import React, { useEffect, useState } from "react";
import { Activity, Cpu, HardDrive } from "lucide-react";

interface SystemMetrics {
  cpu_usage_percent: number;
  ram_used_mb: number;
  ram_total_mb: number;
  vram_used_mb?: number | null;
  vram_total_mb?: number | null;
  gpu_name?: string | null;
}

interface HealthStatus {
  status: string;
  device: string;
  active_provider: string;
  model_loaded: boolean;
  system_metrics?: SystemMetrics | null;
}

export default function Header() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/health");
        if (res.ok && isMounted) {
          const data = await res.json();
          setHealth(data);
        }
      } catch {
        if (isMounted) setHealth(null);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 4000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Left: Branding & University info */}
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-100">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
                ViSEC SER Enterprise v2.0
              </h1>
              <span className="hidden sm:inline-block bg-indigo-50 border border-indigo-200 text-indigo-700 text-[10px] font-bold px-2 py-0.5 rounded-md">
                HCMUTE
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">
              Nhận diện cảm xúc qua giọng nói tiếng Việt (ONNX Runtime)
            </p>
          </div>
        </div>

        {/* Right: Hardware & Health telemetry badges */}
        <div className="flex items-center space-x-2 sm:space-x-3">
          {health ? (
            <>
              {/* Telemetry info chip (hidden on mobile) */}
              <div className="hidden md:flex items-center space-x-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700">
                <Cpu className="w-3.5 h-3.5 text-indigo-600" />
                <span>
                  {health.system_metrics?.gpu_name
                    ? health.system_metrics.gpu_name
                    : health.device}
                </span>
                <span className="text-slate-300">|</span>
                <HardDrive className="w-3.5 h-3.5 text-slate-500" />
                <span>CPU: {health.system_metrics?.cpu_usage_percent ?? 0}%</span>
              </div>

              {/* Status Indicator */}
              <div className="flex items-center space-x-2 bg-emerald-50 border border-emerald-200 px-3 py-1.5 rounded-lg text-xs font-semibold text-emerald-800 shadow-sm">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="truncate max-w-[120px] sm:max-w-none">
                  {health.active_provider === "CUDAExecutionProvider"
                    ? "CUDA GPU Online"
                    : "Hệ thống sẵn sàng"}
                </span>
              </div>
            </>
          ) : (
            <div className="flex items-center space-x-2 bg-amber-50 border border-amber-200 px-3 py-1.5 rounded-lg text-xs font-semibold text-amber-800">
              <span className="w-2 h-2 rounded-full bg-amber-500" />
              <span>Chưa kết nối máy chủ</span>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
