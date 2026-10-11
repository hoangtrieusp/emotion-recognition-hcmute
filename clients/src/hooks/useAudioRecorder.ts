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
    "audio/wav",
    "audio/mp4",
  ];

  for (const mime of mimeCandidates) {
    if (MediaRecorder.isTypeSupported(mime)) {
      return mime;
    }
  }

  return "audio/webm";
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
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up object URL when it changes or on unmount
  useEffect(() => {
    return () => {
      if (audioPreviewUrl) {
        URL.revokeObjectURL(audioPreviewUrl);
      }
    };
  }, [audioPreviewUrl]);

  // Clean up recording tracks & timers on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [audioStream]);

  const resetRecording = useCallback(() => {
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    setRecordedBlob(null);
    setRecordDuration(0);
    setError(null);
    chunksRef.current = [];
  }, [audioPreviewUrl]);

  const startRecording = useCallback(async () => {
    try {
      setError(null);
      resetRecording();

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      setAudioStream(stream);
      setIsRecording(true);
      setRecordDuration(0);

      const supportedMime = getSupportedAudioMimeType();
      setRecordingMimeType(supportedMime);

      const recorder = new MediaRecorder(stream, { mimeType: supportedMime });
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event: BlobEvent) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onstop = () => {
        // Chốt chặn sinh tử 2: Đảm bảo Blob xuất xưởng có đúng MIME Type ('audio/webm' hoặc 'audio/wav')
        const finalType = supportedMime.includes("wav") ? "audio/wav" : "audio/webm";
        const finalBlob = new Blob(chunksRef.current, { type: finalType });
        setRecordedBlob(finalBlob);

        const url = URL.createObjectURL(finalBlob);
        setAudioPreviewUrl(url);
      };

      recorder.start(100); // 100ms chunk slices

      timerRef.current = setInterval(() => {
        setRecordDuration((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError("Không thể truy cập Microphone: " + msg);
      setIsRecording(false);
    }
  }, [resetRecording]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);

      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }

      if (audioStream) {
        audioStream.getTracks().forEach((track) => track.stop());
        setAudioStream(null);
      }
    }
  }, [isRecording, audioStream]);

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
