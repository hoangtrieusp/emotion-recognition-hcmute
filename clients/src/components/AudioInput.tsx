"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, Play, Pause, FileAudio, RefreshCw, Flame, Smile, Meh, Frown, Sparkles } from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

interface AudioInputProps {
  onAnalyze: (file: File | Blob) => void;
  isAnalyzing: boolean;
  isLoading?: boolean;
}

interface BenchmarkSample {
  id: string;
  name: string;
  english: string;
  filename: string;
  icon: React.ReactNode;
  btnStyle: string;
  activeStyle: string;
}

const BENCHMARK_SAMPLES: BenchmarkSample[] = [
  {
    id: "angry",
    name: "Tức giận",
    english: "Angry",
    filename: "angry_sample.wav",
    icon: <Flame className="w-4 h-4 text-rose-600" />,
    btnStyle: "bg-rose-50/70 border-rose-200 text-rose-800 hover:bg-rose-100 hover:border-rose-300",
    activeStyle: "ring-2 ring-rose-500 bg-rose-100 border-rose-400",
  },
  {
    id: "happy",
    name: "Vui vẻ",
    english: "Happy",
    filename: "happy_sample.wav",
    icon: <Smile className="w-4 h-4 text-emerald-600" />,
    btnStyle: "bg-emerald-50/70 border-emerald-200 text-emerald-800 hover:bg-emerald-100 hover:border-emerald-300",
    activeStyle: "ring-2 ring-emerald-500 bg-emerald-100 border-emerald-400",
  },
  {
    id: "neutral",
    name: "Bình thường",
    english: "Neutral",
    filename: "neutral_sample.wav",
    icon: <Meh className="w-4 h-4 text-slate-600" />,
    btnStyle: "bg-slate-100/70 border-slate-200 text-slate-800 hover:bg-slate-200/80 hover:border-slate-300",
    activeStyle: "ring-2 ring-slate-500 bg-slate-200 border-slate-400",
  },
  {
    id: "sad",
    name: "Buồn bã",
    english: "Sad",
    filename: "sad_sample.wav",
    icon: <Frown className="w-4 h-4 text-indigo-600" />,
    btnStyle: "bg-indigo-50/70 border-indigo-200 text-indigo-800 hover:bg-indigo-100 hover:border-indigo-300",
    activeStyle: "ring-2 ring-indigo-500 bg-indigo-100 border-indigo-400",
  },
];

export default function AudioInput({ onAnalyze, isAnalyzing, isLoading }: AudioInputProps) {
  const effectiveAnalyzing = isAnalyzing ?? isLoading ?? false;
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null);

  // Audio Recorder Hook (Web Audio API)
  const {
    isRecording,
    recordDuration,
    recordedBlob,
    audioPreviewUrl: recorderPreviewUrl,
    audioStream,
    error: recorderError,
    startRecording,
    stopRecording,
    resetRecording,
  } = useAudioRecorder();

  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);

  // Audio playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Active audio URL to play preview
  const currentPreviewUrl = selectedFile ? filePreviewUrl : recorderPreviewUrl;

  // Cleanup file preview URL when changing
  useEffect(() => {
    return () => {
      if (filePreviewUrl) {
        URL.revokeObjectURL(filePreviewUrl);
      }
    };
  }, [filePreviewUrl]);

  // Handle local file drop/upload
  const handleFileChange = (file: File) => {
    if (effectiveAnalyzing) return;
    resetRecording();
    setActiveSampleId(null);
    setSelectedFile(file);
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
    setFilePreviewUrl(URL.createObjectURL(file));
  };

  // Handle Benchmark Preset Click
  const handleSelectSample = async (sample: BenchmarkSample) => {
    if (effectiveAnalyzing || isRecording) return;
    try {
      setActiveSampleId(sample.id);
      resetRecording();

      const response = await fetch(`/samples/${sample.filename}`);
      if (!response.ok) {
        throw new Error(`Không thể nạp file mẫu ${sample.filename}`);
      }
      const blob = await response.blob();
      const sampleFile = new File([blob], sample.filename, { type: "audio/wav" });
      
      setSelectedFile(sampleFile);
      if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
      setFilePreviewUrl(URL.createObjectURL(sampleFile));

      // Tự động kích hoạt suy luận ngay lập tức phục vụ demo bảo vệ đồ án
      onAnalyze(sampleFile);
    } catch (err) {
      console.error("Lỗi nạp mẫu âm thanh:", err);
      setActiveSampleId(null);
    }
  };

  // Handle Play/Pause
  const togglePlayAudio = () => {
    const audio = audioPlayerRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play().then(() => setIsPlaying(true)).catch(() => setIsPlaying(false));
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  // Trigger analysis with lock
  const handleSubmitAnalysis = () => {
    if (effectiveAnalyzing) return;
    if (selectedFile) {
      onAnalyze(selectedFile);
    } else if (recordedBlob) {
      onAnalyze(recordedBlob);
    }
  };

  const hasAudioReady = Boolean(selectedFile || recordedBlob);

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      {/* Tab selection */}
      <div className="flex items-center space-x-2 border-b border-slate-200 pb-4 mb-5">
        <button
          type="button"
          disabled={effectiveAnalyzing || isRecording}
          onClick={() => {
            setActiveTab("mic");
            setActiveSampleId(null);
          }}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === "mic"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-100"
              : "text-slate-600 hover:bg-slate-100"
          } ${effectiveAnalyzing || isRecording ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <Mic className="w-4 h-4" />
          <span>Thu âm Microphone</span>
        </button>
        <button
          type="button"
          disabled={effectiveAnalyzing || isRecording}
          onClick={() => {
            setActiveTab("upload");
            setActiveSampleId(null);
          }}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === "upload"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-100"
              : "text-slate-600 hover:bg-slate-100"
          } ${effectiveAnalyzing || isRecording ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <Upload className="w-4 h-4" />
          <span>Tải file âm thanh</span>
        </button>
      </div>

      {/* Mic Tab Content */}
      {activeTab === "mic" && (
        <div className="space-y-4">
          <WaveformVisualizer isRecording={isRecording} stream={audioStream} />

          {recorderError && (
            <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 p-3 rounded-xl">
              {recorderError}
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-3">
              {!isRecording ? (
                <button
                  type="button"
                  onClick={startRecording}
                  disabled={effectiveAnalyzing}
                  className={`flex items-center space-x-2 bg-rose-600 hover:bg-rose-700 text-white px-5 py-2.5 rounded-xl font-semibold text-sm shadow-sm transition-all ${
                    effectiveAnalyzing ? "opacity-50 cursor-not-allowed pointer-events-none" : ""
                  }`}
                >
                  <Mic className="w-4 h-4" />
                  <span>Bắt đầu thu âm</span>
                </button>
              ) : (
                <button
                  type="button"
                  onClick={stopRecording}
                  className="flex items-center space-x-2 bg-slate-800 hover:bg-slate-900 text-white px-5 py-2.5 rounded-xl font-semibold text-sm shadow-sm transition-all animate-pulse"
                >
                  <Square className="w-4 h-4" />
                  <span>Dừng thu âm ({recordDuration}s)</span>
                </button>
              )}

              {isRecording && (
                <div className="flex items-center space-x-2 text-rose-600 text-sm font-medium animate-pulse">
                  <span className="w-2.5 h-2.5 rounded-full bg-rose-600" />
                  <span>Đang ghi âm...</span>
                </div>
              )}
            </div>

            {recordedBlob && !isRecording && (
              <span className="text-xs text-emerald-700 font-semibold bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200">
                ✓ Đã ghi âm xong ({recordDuration}s)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Upload Tab Content */}
      {activeTab === "upload" && (
        <div className="space-y-4">
          <label className={`border-2 border-dashed border-slate-300 hover:border-indigo-500 bg-slate-50 hover:bg-indigo-50/30 rounded-2xl p-8 flex flex-col items-center justify-center transition-all ${
            effectiveAnalyzing ? "opacity-50 cursor-not-allowed pointer-events-none" : "cursor-pointer"
          }`}>
            <FileAudio className="w-10 h-10 text-slate-400 mb-2" />
            <p className="text-sm font-semibold text-slate-700">
              Kéo thả hoặc nhấp để chọn file âm thanh
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Hỗ trợ WAV, MP3, M4A, OGG, WEBM, FLAC
            </p>
            <input
              type="file"
              disabled={effectiveAnalyzing}
              accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg,.flac"
              onChange={(e) => {
                if (effectiveAnalyzing) return;
                if (e.target.files && e.target.files[0]) {
                  handleFileChange(e.target.files[0]);
                }
              }}
              className="hidden"
            />
          </label>

          {selectedFile && (
            <div className="flex items-center justify-between bg-slate-50 border border-slate-200 px-4 py-2.5 rounded-xl text-xs">
              <span className="font-semibold text-slate-800 truncate max-w-xs">
                📄 {selectedFile.name}
              </span>
              <span className="text-slate-500">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>
          )}
        </div>
      )}

      {/* 4 Nút Mẫu Test Nhanh (Preset Benchmark Buttons) */}
      <div className="mt-6 pt-5 border-t border-slate-100">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-1.5 text-xs font-bold uppercase tracking-wider text-slate-600">
            <Sparkles className="w-3.5 h-3.5 text-indigo-600" />
            <span>Mẫu thử nhanh (Benchmark Presets):</span>
          </div>
          <span className="text-[11px] text-slate-400 font-medium hidden sm:inline">
            Tự động nạp mẫu & phân tích 1 chạm
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          {BENCHMARK_SAMPLES.map((sample) => {
            const isActive = activeSampleId === sample.id;
            return (
              <button
                key={sample.id}
                type="button"
                disabled={effectiveAnalyzing || isRecording}
                onClick={() => handleSelectSample(sample)}
                className={`flex items-center justify-center space-x-2 py-2.5 px-3 rounded-xl border text-xs font-bold transition-all shadow-sm ${
                  sample.btnStyle
                } ${isActive ? sample.activeStyle : ""} ${
                  effectiveAnalyzing || isRecording ? "opacity-50 cursor-not-allowed" : "hover:scale-[1.02] active:scale-[0.98]"
                }`}
              >
                {sample.icon}
                <div className="text-left">
                  <div className="leading-tight">{sample.name}</div>
                  <div className="text-[10px] opacity-70 font-medium">({sample.english})</div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Audio Playback Preview & Submit Button */}
      {currentPreviewUrl && (
        <div className="mt-6 pt-5 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={togglePlayAudio}
              className="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-900 text-white flex items-center justify-center shadow-sm shrink-0 transition-transform active:scale-95"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <div className="text-left">
              <span className="text-xs font-bold text-slate-700 block">
                Nghe lại đoạn âm thanh
              </span>
              <span className="text-[11px] text-slate-500">
                {selectedFile ? selectedFile.name : `Microphone (${recordDuration}s)`}
              </span>
            </div>
            <audio
              ref={audioPlayerRef}
              src={currentPreviewUrl}
              onEnded={handleAudioEnded}
              className="hidden"
            />
          </div>

          <button
            type="button"
            onClick={handleSubmitAnalysis}
            disabled={!hasAudioReady || effectiveAnalyzing}
            className={`w-full sm:w-auto flex items-center justify-center space-x-2 px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all ${
              !hasAudioReady || effectiveAnalyzing
                ? "bg-slate-200 text-slate-400 cursor-not-allowed pointer-events-none"
                : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:shadow-lg active:scale-95"
            }`}
          >
            {effectiveAnalyzing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>ĐANG PHÂN TÍCH...</span>
              </>
            ) : (
              <>
                <span>PHÂN TÍCH CẢM XÚC</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
