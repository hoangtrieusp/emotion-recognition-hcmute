import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional
import psutil
import numpy as np

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from servers.app.schemas import EmotionPredictionResponse, HealthResponse, Telemetry, SystemMetrics
    from servers.app.core.audio_processor import preprocess_audio_pipeline, aggregate_salience_probabilities
    from servers.app.engine.onnx_engine import ONNXInferenceEngine, EMOTION_MAPPING
except ImportError:
    from app.schemas import EmotionPredictionResponse, HealthResponse, Telemetry, SystemMetrics
    from app.core.audio_processor import preprocess_audio_pipeline, aggregate_salience_probabilities
    from app.engine.onnx_engine import ONNXInferenceEngine, EMOTION_MAPPING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ser_api")

engine: Optional[ONNXInferenceEngine] = None

def collect_system_metrics() -> SystemMetrics:
    cpu_pct = float(psutil.cpu_percent(interval=None))
    mem = psutil.virtual_memory()
    ram_used_mb = round(mem.used / (1024 * 1024), 1)
    ram_total_mb = round(mem.total / (1024 * 1024), 1)

    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    gpu_name: Optional[str] = None

    try:
        import torch
        if torch.cuda.is_available():
            vram_used_mb = round(torch.cuda.memory_allocated() / (1024 * 1024), 1)
            props = torch.cuda.get_device_properties(0)
            vram_total_mb = round(props.total_memory / (1024 * 1024), 1)
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as e:
        logger.debug(f"Không thể truy xuất thông tin CUDA VRAM: {e}")

    return SystemMetrics(
        cpu_usage_percent=cpu_pct,
        ram_used_mb=ram_used_mb,
        ram_total_mb=ram_total_mb,
        vram_used_mb=vram_used_mb,
        vram_total_mb=vram_total_mb,
        gpu_name=gpu_name,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Khởi động hệ thống ViSEC SER Enterprise v2.0...")
    engine = ONNXInferenceEngine.get_instance()
    logger.info(f"Engine sẵn sàng phục vụ trên: {engine.device_name} (Provider: {engine.active_provider})")
    psutil.cpu_percent(interval=None)
    yield
    logger.info("Đang tắt hệ thống ViSEC SER Enterprise...")

app = FastAPI(
    title="ViSEC SER Enterprise Engine",
    description="High-Performance Speech Emotion Recognition API (FastAPI + ONNX Runtime Dual-Engine)",
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:[0-9]+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    metrics = collect_system_metrics()
    if engine is None:
        return HealthResponse(
            status="starting",
            device="unknown",
            active_provider="none",
            model_loaded=False,
            system_metrics=metrics
        )
    status_info = engine.get_status()
    return HealthResponse(
        status="online",
        device=status_info["device"],
        active_provider=status_info["active_provider"],
        model_loaded=True,
        system_metrics=metrics
    )

@app.post("/api/v1/predict/emotion", response_model=EmotionPredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Mô hình AI chưa sẵn sàng hoặc đang khởi động.")

    total_start = time.perf_counter()
    
    try:
        try:
            audio_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể đọc file âm thanh: {str(e)}")

        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="File âm thanh rỗng.")

        dsp_start = time.perf_counter()
        try:
            dsp_result = preprocess_audio_pipeline(audio_bytes)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lỗi tiền xử lý tín hiệu: {str(e)}")
        dsp_duration_ms = (time.perf_counter() - dsp_start) * 1000.0

        if not dsp_result["is_valid"]:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "warning",
                    "error": dsp_result["error"],
                    "dominant_emotion": None,
                    "dominant_probability": None,
                    "probabilities": None,
                    "telemetry": {
                        "duration": round(dsp_result["duration"], 2),
                        "rms": round(dsp_result["rms"], 5),
                        "num_chunks": 0,
                        "dsp_latency_ms": round(dsp_duration_ms, 2),
                        "inference_latency_ms": 0.0,
                        "total_latency_ms": round(dsp_duration_ms, 2)
                    }
                }
            )

        infer_start = time.perf_counter()
        chunks = dsp_result["chunks"]
        chunk_probs = engine.predict_chunks(chunks)
        infer_duration_ms = (time.perf_counter() - infer_start) * 1000.0

        final_probs = aggregate_salience_probabilities(chunk_probs, dsp_result.get("chunk_weights"))
        
        probabilities = {
            EMOTION_MAPPING[idx]: round(float(final_probs[idx]), 4)
            for idx in range(4)
        }
        
        dominant_idx = int(np.argmax(final_probs))
        dominant_emotion = EMOTION_MAPPING[dominant_idx]
        dominant_prob = round(float(final_probs[dominant_idx]), 4)

        total_duration_ms = (time.perf_counter() - total_start) * 1000.0

        telemetry = Telemetry(
            duration=round(dsp_result["duration"], 2),
            rms=round(dsp_result["rms"], 5),
            num_chunks=dsp_result["num_chunks"],
            dsp_latency_ms=round(dsp_duration_ms, 2),
            inference_latency_ms=round(infer_duration_ms, 2),
            total_latency_ms=round(total_duration_ms, 2)
        )

        return EmotionPredictionResponse(
            status="success",
            dominant_emotion=dominant_emotion,
            dominant_probability=dominant_prob,
            probabilities=probabilities,
            telemetry=telemetry,
            error=None
        )
    finally:
        await file.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("servers.app.main:app", host="127.0.0.1", port=8000, reload=True)
