"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, Play, Pause, FileAudio, RefreshCw, AlertCircle } from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer";
import { useAudioRecorder } from "../hooks/useAudioRecorder";

interface AudioInputProps {
  onAnalyze: (file: File | Blob) => void;
  isAnalyzing: boolean;
  isLoading?: boolean;
}

export default function AudioInput({ onAnalyze, isAnalyzing, isLoading }: AudioInputProps) {
  const effectiveAnalyzing = isAnalyzing ?? isLoading ?? false;
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");

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

  // File upload state (Riêng biệt cho Tab Upload)
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);

  // Audio playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);

  // Cách ly hoàn toàn State giữa 2 Tab:
  // - Tab Mic: dùng recorderPreviewUrl và recordedBlob
  // - Tab Upload: dùng filePreviewUrl và selectedFile
  const currentPreviewUrl = activeTab === "mic" ? recorderPreviewUrl : filePreviewUrl;
  const currentAudioName =
    activeTab === "mic"
      ? `Bản thu âm Microphone (${recordDuration}s)`
      : selectedFile?.name || "File tải lên";

  const hasAudioReady =
    activeTab === "mic" ? Boolean(recordedBlob) : Boolean(selectedFile);

  // Cleanup file preview URL khi đổi file hoặc unmount
  useEffect(() => {
    return () => {
      if (filePreviewUrl) {
        URL.revokeObjectURL(filePreviewUrl);
      }
    };
  }, [filePreviewUrl]);

  // Đổi preview URL thì reset trạng thái phát nhạc
  useEffect(() => {
    setIsPlaying(false);
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
      audioPlayerRef.current.currentTime = 0;
    }
  }, [currentPreviewUrl]);

  // Chuyển Tab: Dừng phát nhạc để tránh xung đột
  const handleSwitchTab = (tab: "mic" | "upload") => {
    if (effectiveAnalyzing || isRecording) return;
    setIsPlaying(false);
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }
    setActiveTab(tab);
  };

  // Xử lý chọn file từ ổ đĩa
  const handleFileChange = (file: File) => {
    if (effectiveAnalyzing) return;
    setSelectedFile(file);
    if (filePreviewUrl) URL.revokeObjectURL(filePreviewUrl);
    setFilePreviewUrl(URL.createObjectURL(file));
  };

  // Bắt đầu thu âm: Dừng nhạc và gọi hook
  const handleStartRecording = async () => {
    if (effectiveAnalyzing) return;
    setIsPlaying(false);
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause();
    }
    await startRecording();
  };

  // Điều khiển Phát/Dừng nghe lại
  const togglePlayAudio = () => {
    const audio = audioPlayerRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  // Gửi âm thanh đi phân tích: Đúng tab nào gửi dữ liệu tab đó
  const handleSubmitAnalysis = () => {
    if (effectiveAnalyzing) return;
    if (activeTab === "mic" && recordedBlob) {
      onAnalyze(recordedBlob);
    } else if (activeTab === "upload" && selectedFile) {
      onAnalyze(selectedFile);
    }
  };

  return (
    <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
      {/* Tab selection */}
      <div className="flex items-center space-x-2 border-b border-slate-200 pb-4 mb-5">
        <button
          type="button"
          disabled={effectiveAnalyzing || isRecording}
          onClick={() => handleSwitchTab("mic")}
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
          onClick={() => handleSwitchTab("upload")}
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
            <div className="text-xs text-rose-600 bg-rose-50 border border-rose-200 p-3 rounded-xl flex items-center space-x-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{recorderError}</span>
            </div>
          )}

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-3">
              {!isRecording ? (
                <button
                  type="button"
                  onClick={handleStartRecording}
                  disabled={effectiveAnalyzing}
                  className={`flex items-center space-x-2 bg-rose-600 hover:bg-rose-700 text-white px-5 py-2.5 rounded-xl font-semibold text-sm shadow-sm transition-all ${
                    effectiveAnalyzing ? "opacity-50 cursor-not-allowed pointer-events-none" : ""
                  }`}
                >
                  <Mic className="w-4 h-4" />
                  <span>{recordedBlob ? "Thu âm lại" : "Bắt đầu thu âm"}</span>
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
                  <span>Đang ghi âm giọng nói của bạn...</span>
                </div>
              )}
            </div>

            {recordedBlob && !isRecording && (
              <span className="text-xs text-emerald-700 font-semibold bg-emerald-50 px-3 py-1 rounded-full border border-emerald-200">
                ✓ Đã thu âm xong ({recordDuration}s)
              </span>
            )}
          </div>
        </div>
      )}

      {/* Upload Tab Content */}
      {activeTab === "upload" && (
        <div className="space-y-4">
          <label
            className={`border-2 border-dashed border-slate-300 hover:border-indigo-500 bg-slate-50 hover:bg-indigo-50/30 rounded-2xl p-8 flex flex-col items-center justify-center transition-all ${
              effectiveAnalyzing ? "opacity-50 cursor-not-allowed pointer-events-none" : "cursor-pointer"
            }`}
          >
            <FileAudio className="w-10 h-10 text-slate-400 mb-2" />
            <p className="text-sm font-semibold text-slate-700">
              Kéo thả hoặc nhấp để chọn file âm thanh
            </p>
            <p className="text-xs text-slate-500 mt-1">Hỗ trợ WAV, MP3, M4A, OGG, WEBM, FLAC</p>
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
              <span className="text-[11px] text-slate-500">{currentAudioName}</span>
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
