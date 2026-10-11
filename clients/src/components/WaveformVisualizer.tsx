"use client";

import React, { useEffect, useRef } from "react";

interface WaveformVisualizerProps {
  isRecording: boolean;
  stream: MediaStream | null;
}

export default function WaveformVisualizer({ isRecording, stream }: WaveformVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationFrameId: number;
    let audioCtx: AudioContext | null = null;
    let analyser: AnalyserNode | null = null;
    let dataArray: Uint8Array | null = null;

    if (isRecording && stream) {
      try {
        audioCtx = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        const bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(new ArrayBuffer(bufferLength));
        source.connect(analyser);
      } catch (err) {
        console.error("Web Audio API error:", err);
      }
    }

    const render = () => {
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);

      // Background gradient subtle
      const bgGrad = ctx.createLinearGradient(0, 0, width, 0);
      bgGrad.addColorStop(0, "#f1f5f9");
      bgGrad.addColorStop(1, "#f8fafc");
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      if (isRecording && analyser && dataArray) {
        analyser.getByteFrequencyData(dataArray as unknown as Uint8Array<ArrayBuffer>);

        const barWidth = (width / dataArray.length) * 2.2;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
          const barHeight = (dataArray[i] / 255) * (height * 0.85);

          // Bar color with vibrant indigo-to-purple gradient
          const grad = ctx.createLinearGradient(0, height - barHeight, 0, height);
          grad.addColorStop(0, "#4f46e5"); // Indigo 600
          grad.addColorStop(1, "#818cf8"); // Indigo 400

          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.roundRect(x, (height - barHeight) / 2, barWidth - 1.5, Math.max(3, barHeight), 2);
          ctx.fill();

          x += barWidth;
        }
      } else {
        // Resting wave animation
        ctx.beginPath();
        ctx.strokeStyle = "#cbd5e1";
        ctx.lineWidth = 2;
        ctx.moveTo(0, height / 2);
        for (let x = 0; x < width; x += 10) {
          const y = height / 2 + Math.sin(x * 0.05 + Date.now() * 0.002) * 3;
          ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      if (audioCtx) {
        audioCtx.close().catch(() => {});
      }
    };
  }, [isRecording, stream]);

  return (
    <div className="w-full bg-slate-100 rounded-xl overflow-hidden border border-slate-200 shadow-inner">
      <canvas
        ref={canvasRef}
        width={600}
        height={100}
        className="w-full h-24 block"
      />
    </div>
  );
}
