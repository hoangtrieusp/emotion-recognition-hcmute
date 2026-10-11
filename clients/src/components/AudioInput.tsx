"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, Play, Pause, FileAudio, Flame, Smile, Meh, Frown, RefreshCw, Volume2 } from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

interface AudioInputProps {
  onAnalyze: (file: File | Blob) => void;
  isAnalyzing: boolean;
  isLoading?: boolean; // For backward compatibility
}

export default function AudioInput({ onAnalyze, isAnalyzing, isLoading }: AudioInputProps) {
  const effectiveAnalyzing = isAnalyzing ?? isLoading ?? false;
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");

  // Audio Recorder Hook (Strict Blob MIME Type enforcement)
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

  // File upload state & preset state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPresetName, setSelectedPresetName] = useState<string | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const [isLoadingPreset, setIsLoadingPreset] = useState(false);

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
    setSelectedFile(file);
    setSelectedPresetName(null);
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
    setFilePreviewUrl(URL.createObjectURL(file));
  };

  // Load preset sample with race-condition lock (Chốt chặn 3)
  const handleSelectPreset = async (emotionKey: string, label: string) => {
    if (effectiveAnalyzing || isRecording || isLoadingPreset) return;
    setIsLoadingPreset(true);
    try {
      setSelectedPresetName(label);
      resetRecording();
      const res = await fetch(`/samples/${emotionKey}_sample.wav`);
      if (!res.ok) throw new Error("Không tìm thấy file mẫu");
      const blob = await res.blob();
      const file = new File([blob], `${emotionKey}_sample.wav`, { type: "audio/wav" });
      setSelectedFile(file);
      if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
      setFilePreviewUrl(URL.createObjectURL(file));
    } catch (err) {
      alert("Lỗi tải file mẫu: " + err);
    } finally {
      setIsLoadingPreset(false);
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
          onClick={() => setActiveTab("mic")}
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
          onClick={() => setActiveTab("upload")}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === "upload"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-100"
              : "text-slate-600 hover:bg-slate-100"
          } ${effectiveAnalyzing || isRecording ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          <Upload className="w-4 h-4" />
          <span>Tải file Âm thanh</span>
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
                ✓ Đã ghi âm xong ({recordDuration}s · {recordedBlob.type})
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
              Hỗ trợ WAV, MP3, M4A, OGG, WEBM (Tối đa 25MB)
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

      {/* Preset Quick Test Section (Chốt chặn 3: Khóa tương tác khi isAnalyzing hoặc isLoadingPreset) */}
      <div className="mt-6 pt-5 border-t border-slate-100">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center space-x-1.5">
            <Volume2 className="w-3.5 h-3.5 text-slate-400" />
            <span>Thử nhanh bằng File mẫu (Benchmark Samples):</span>
          </span>
          {isLoadingPreset ? (
            <span className="text-xs text-indigo-600 font-semibold flex items-center space-x-1">
              <RefreshCw className="w-3 h-3 animate-spin" />
              <span>Đang tải file mẫu...</span>
            </span>
          ) : selectedPresetName ? (
            <span className="text-xs text-indigo-600 font-semibold">
              Đang chọn: {selectedPresetName}
            </span>
          ) : null}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <button
            type="button"
            disabled={effectiveAnalyzing || isRecording || isLoadingPreset}
            onClick={() => handleSelectPreset("angry", "Tức giận (Angry)")}
            className={`flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 text-xs font-semibold transition-all ${
              effectiveAnalyzing || isRecording || isLoadingPreset
                ? "opacity-40 cursor-not-allowed pointer-events-none"
                : ""
            }`}
          >
            <Flame className="w-3.5 h-3.5 text-rose-600" />
            <span>Tức giận (Angry)</span>
          </button>
          <button
            type="button"
            disabled={effectiveAnalyzing || isRecording || isLoadingPreset}
            onClick={() => handleSelectPreset("happy", "Vui vẻ (Happy)")}
            className={`flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 text-xs font-semibold transition-all ${
              effectiveAnalyzing || isRecording || isLoadingPreset
                ? "opacity-40 cursor-not-allowed pointer-events-none"
                : ""
            }`}
          >
            <Smile className="w-3.5 h-3.5 text-emerald-600" />
            <span>Vui vẻ (Happy)</span>
          </button>
          <button
            type="button"
            disabled={effectiveAnalyzing || isRecording || isLoadingPreset}
            onClick={() => handleSelectPreset("neutral", "Bình thường (Neutral)")}
            className={`flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 text-xs font-semibold transition-all ${
              effectiveAnalyzing || isRecording || isLoadingPreset
                ? "opacity-40 cursor-not-allowed pointer-events-none"
                : ""
            }`}
          >
            <Meh className="w-3.5 h-3.5 text-slate-600" />
            <span>Bình thường</span>
          </button>
          <button
            type="button"
            disabled={effectiveAnalyzing || isRecording || isLoadingPreset}
            onClick={() => handleSelectPreset("sad", "Buồn bã (Sad)")}
            className={`flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 text-xs font-semibold transition-all ${
              effectiveAnalyzing || isRecording || isLoadingPreset
                ? "opacity-40 cursor-not-allowed pointer-events-none"
                : ""
            }`}
          >
            <Frown className="w-3.5 h-3.5 text-indigo-600" />
            <span>Buồn bã (Sad)</span>
          </button>
        </div>
      </div>

      {/* Audio Playback Preview & Submit Button */}
      {currentPreviewUrl && (
        <div className="mt-6 pt-5 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={togglePlayAudio}
              className="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-900 text-white flex items-center justify-center shadow-sm shrink-0"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <span className="text-xs font-semibold text-slate-600">
              Nghe lại đoạn âm thanh đã chuẩn bị
            </span>
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
                : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:shadow-lg"
            }`}
          >
            {effectiveAnalyzing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Đang xử lý & Suy luận ONNX...</span>
              </>
            ) : (
              <>
                <span>🚀 PHÂN TÍCH CẢM XÚC</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}
