"use client";

import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, Upload, Play, Pause, FileAudio, Flame, Smile, Meh, Frown, RefreshCw } from "lucide-react";
import WaveformVisualizer from "./WaveformVisualizer";

interface AudioInputProps {
  onAnalyze: (file: File | Blob) => void;
  isLoading: boolean;
}

export default function AudioInput({ onAnalyze, isLoading }: AudioInputProps) {
  const [activeTab, setActiveTab] = useState<"mic" | "upload">("mic");
  
  // Microphone recording states
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  
  // File upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPresetName, setSelectedPresetName] = useState<string | null>(null);
  
  // Audio playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const audioPlayerRef = useRef<HTMLAudioElement | null>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  // Cleanup audio preview URL when changing
  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  // Start microphone recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setAudioStream(stream);
      setIsRecording(true);
      setRecordDuration(0);
      setRecordedBlob(null);
      setSelectedFile(null);
      setSelectedPresetName(null);
      if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);

      const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/ogg";
      const recorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecordedBlob(blob);
        const url = URL.createObjectURL(blob);
        setAudioPreviewUrl(url);
      };

      recorder.start(100);

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err) {
      alert("Không thể truy cập Microphone. Vui lòng cấp quyền trong trình duyệt: " + err);
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
      if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
        setAudioStream(null);
      }
    }
  };

  // Handle local file drop/upload
  const handleFileChange = (file: File) => {
    setSelectedFile(file);
    setRecordedBlob(null);
    setSelectedPresetName(null);
    if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
    setAudioPreviewUrl(URL.createObjectURL(file));
  };

  // Load preset sample
  const handleSelectPreset = async (emotionKey: string, label: string) => {
    try {
      setSelectedPresetName(label);
      setRecordedBlob(null);
      const res = await fetch(`/samples/${emotionKey}_sample.wav`);
      if (!res.ok) throw new Error("Không tìm thấy file mẫu");
      const blob = await res.blob();
      const file = new File([blob], `${emotionKey}_sample.wav`, { type: "audio/wav" });
      setSelectedFile(file);
      if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(URL.createObjectURL(file));
    } catch (err) {
      alert("Lỗi tải file mẫu: " + err);
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

  // Trigger analysis
  const handleSubmitAnalysis = () => {
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
          onClick={() => setActiveTab("mic")}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === "mic"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-100"
              : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Mic className="w-4 h-4" />
          <span>Thu âm Microphone</span>
        </button>
        <button
          onClick={() => setActiveTab("upload")}
          className={`flex items-center space-x-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all ${
            activeTab === "upload"
              ? "bg-indigo-600 text-white shadow-sm shadow-indigo-100"
              : "text-slate-600 hover:bg-slate-100"
          }`}
        >
          <Upload className="w-4 h-4" />
          <span>Tải file Âm thanh</span>
        </button>
      </div>

      {/* Mic Tab Content */}
      {activeTab === "mic" && (
        <div className="space-y-4">
          <WaveformVisualizer isRecording={isRecording} stream={audioStream} />

          <div className="flex items-center justify-between pt-2">
            <div className="flex items-center space-x-3">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  disabled={isLoading}
                  className="flex items-center space-x-2 bg-rose-600 hover:bg-rose-700 text-white px-5 py-2.5 rounded-xl font-semibold text-sm shadow-sm transition-all"
                >
                  <Mic className="w-4 h-4" />
                  <span>Bắt đầu thu âm</span>
                </button>
              ) : (
                <button
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
          <label className="border-2 border-dashed border-slate-300 hover:border-indigo-500 bg-slate-50 hover:bg-indigo-50/30 rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all">
            <FileAudio className="w-10 h-10 text-slate-400 mb-2" />
            <p className="text-sm font-semibold text-slate-700">
              Kéo thả hoặc nhấp để chọn file âm thanh
            </p>
            <p className="text-xs text-slate-500 mt-1">
              Hỗ trợ WAV, MP3, M4A, OGG, WEBM (Tối đa 25MB)
            </p>
            <input
              type="file"
              accept="audio/*,.wav,.mp3,.m4a,.webm,.ogg,.flac"
              onChange={(e) => {
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

      {/* Preset Quick Test Section */}
      <div className="mt-6 pt-5 border-t border-slate-100">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
            Thử nhanh bằng File mẫu (Preset Samples):
          </span>
          {selectedPresetName && (
            <span className="text-xs text-indigo-600 font-semibold">
              Đang chọn: {selectedPresetName}
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <button
            type="button"
            onClick={() => handleSelectPreset("angry", "Tức giận (Angry)")}
            className="flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-rose-50 hover:bg-rose-100 text-rose-700 border border-rose-200 text-xs font-semibold transition-colors"
          >
            <Flame className="w-3.5 h-3.5 text-rose-600" />
            <span>Tức giận (Angry)</span>
          </button>
          <button
            type="button"
            onClick={() => handleSelectPreset("happy", "Vui vẻ (Happy)")}
            className="flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border border-emerald-200 text-xs font-semibold transition-colors"
          >
            <Smile className="w-3.5 h-3.5 text-emerald-600" />
            <span>Vui vẻ (Happy)</span>
          </button>
          <button
            type="button"
            onClick={() => handleSelectPreset("neutral", "Bình thường (Neutral)")}
            className="flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 text-xs font-semibold transition-colors"
          >
            <Meh className="w-3.5 h-3.5 text-slate-600" />
            <span>Bình thường</span>
          </button>
          <button
            type="button"
            onClick={() => handleSelectPreset("sad", "Buồn bã (Sad)")}
            className="flex items-center justify-center space-x-1.5 px-3 py-2 rounded-xl bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 text-xs font-semibold transition-colors"
          >
            <Frown className="w-3.5 h-3.5 text-indigo-600" />
            <span>Buồn bã (Sad)</span>
          </button>
        </div>
      </div>

      {/* Audio Playback Preview & Submit Button */}
      {audioPreviewUrl && (
        <div className="mt-6 pt-5 border-t border-slate-100 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <button
              type="button"
              onClick={togglePlayAudio}
              className="w-10 h-10 rounded-full bg-slate-800 hover:bg-slate-900 text-white flex items-center justify-center shadow-sm"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
            </button>
            <span className="text-xs font-semibold text-slate-600">
              Nghe lại đoạn âm thanh đã chuẩn bị
            </span>
            <audio
              ref={audioPlayerRef}
              src={audioPreviewUrl}
              onEnded={handleAudioEnded}
              className="hidden"
            />
          </div>

          <button
            type="button"
            onClick={handleSubmitAnalysis}
            disabled={!hasAudioReady || isLoading}
            className={`w-full sm:w-auto flex items-center justify-center space-x-2 px-6 py-3 rounded-xl font-bold text-sm shadow-md transition-all ${
              !hasAudioReady || isLoading
                ? "bg-slate-200 text-slate-400 cursor-not-allowed"
                : "bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-200 hover:shadow-lg"
            }`}
          >
            {isLoading ? (
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
