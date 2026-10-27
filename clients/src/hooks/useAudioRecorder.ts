"use client";

import { useState, useRef, useEffect, useCallback } from "react";

export interface UseAudioRecorderReturn {
  isRecording: boolean;
  recordDuration: number;
  recordedBlob: Blob | null;
  audioPreviewUrl: string | null;
  audioStream: MediaStream | null;
  recordingMimeType: string;
  error: string | null;
  startRecording: () => Promise<void>;
  stopRecording: () => void;
  resetRecording: () => void;
}

/**
 * Determine supported audio MIME type across modern browsers (Chrome, Edge, Firefox, Safari).
 */
function getSupportedAudioMimeType(): string {
  if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
    return "audio/webm";
  }

  const mimeCandidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
    "audio/wav",
  ];

  for (const mime of mimeCandidates) {
    try {
      if (MediaRecorder.isTypeSupported(mime)) {
        return mime;
      }
    } catch {
      // Ignore unsupported check errors
    }
  }

  return "";
}

export function useAudioRecorder(): UseAudioRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [recordDuration, setRecordDuration] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [audioStream, setAudioStream] = useState<MediaStream | null>(null);
  const [recordingMimeType, setRecordingMimeType] = useState<string>("audio/webm");
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);
  const previewUrlRef = useRef<string | null>(null);

  // Sync ref with state for preview URL cleanup
  useEffect(() => {
    previewUrlRef.current = audioPreviewUrl;
  }, [audioPreviewUrl]);

  // Cleanup on unmount ONLY (empty dependency array to prevent cancelling active recordings)
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        try {
          mediaRecorderRef.current.stop();
        } catch {}
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      if (previewUrlRef.current) {
        URL.revokeObjectURL(previewUrlRef.current);
      }
    };
  }, []);

  const resetRecording = useCallback(() => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = null;
    }
    setAudioPreviewUrl(null);
    setRecordedBlob(null);
    setRecordDuration(0);
    setError(null);
    chunksRef.current = [];
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      resetRecording();

      // Dừng bất kỳ stream cũ nào còn sót
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      streamRef.current = stream;
      setAudioStream(stream);

      const supportedMime = getSupportedAudioMimeType();
      setRecordingMimeType(supportedMime || "audio/webm");

      const options: MediaRecorderOptions = supportedMime ? { mimeType: supportedMime } : {};
      const recorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        const mime = recorder.mimeType || supportedMime || "audio/webm";
        const cleanType = mime.includes("wav") ? "audio/wav" : mime.includes("mp4") ? "audio/mp4" : "audio/webm";
        const finalBlob = new Blob(chunksRef.current, { type: cleanType });
        
        setRecordedBlob(finalBlob);

        const url = URL.createObjectURL(finalBlob);
        setAudioPreviewUrl(url);

        // Dừng các track audio sau khi đã xuất xưởng Blob an toàn
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
        }
        setAudioStream(null);
      };

      recorder.start(100); // 100ms time slice
      setIsRecording(true);
      startTimeRef.current = Date.now();
      setRecordDuration(0);

      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = setInterval(() => {
        const elapsedSec = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setRecordDuration(elapsedSec);
      }, 500);
    } catch (err: unknown) {
      console.error("Microphone error:", err);
      const msg = err instanceof Error ? err.message : String(err);
      setError("Không thể truy cập Microphone: " + msg);
      setIsRecording(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  }, [resetRecording]);

  const stopRecording = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      try {
        mediaRecorderRef.current.stop();
      } catch (err) {
        console.error("Error stopping MediaRecorder:", err);
      }
    }

    setIsRecording(false);

    // Tính toán thời lượng chính xác
    if (startTimeRef.current > 0) {
      const totalSec = Math.max(1, Math.round((Date.now() - startTimeRef.current) / 1000));
      setRecordDuration(totalSec);
    }
  }, []);

  return {
    isRecording,
    recordDuration,
    recordedBlob,
    audioPreviewUrl,
    audioStream,
    recordingMimeType,
    error,
    startRecording,
    stopRecording,
    resetRecording,
  };
}
