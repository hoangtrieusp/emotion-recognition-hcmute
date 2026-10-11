"use client";

import React, { useEffect, useState } from "react";
import { Activity } from "lucide-react";

interface HealthStatus {
  status: string;
  device: string;
  active_provider: string;
  model_loaded: boolean;
}

export default function Header() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/health");
        if (res.ok) {
          const data = await res.json();
          setHealth(data);
        }
      } catch {
        setHealth(null);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 text-white flex items-center justify-center shadow-md shadow-indigo-100">
            <Activity className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-slate-900 tracking-tight">
              Hệ Thống Nhận Diện Cảm Xúc
            </h1>
            <p className="text-xs text-slate-500 font-medium">
              Nhận diện cảm xúc qua giọng nói tiếng Việt
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          {health ? (
            <div className="flex items-center space-x-2 bg-emerald-50 border border-emerald-200 px-3 py-1.5 rounded-lg text-xs font-semibold text-emerald-800 shadow-sm">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>Hệ thống sẵn sàng</span>
            </div>
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
