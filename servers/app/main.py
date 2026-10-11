import os
import time
import logging
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from servers.app.schemas import EmotionPredictionResponse, HealthResponse, Telemetry
from servers.app.core.audio_processor import preprocess_audio_pipeline, aggregate_salience_probabilities
from servers.app.engine.onnx_engine import ONNXInferenceEngine, EMOTION_MAPPING

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ser_api")

engine: ONNXInferenceEngine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Initializing SER Engine Enterprise v2.0...")
    model_path = os.path.join(os.getcwd(), "servers", "weights", "ser_model.onnx")
    engine = ONNXInferenceEngine.get_instance(model_path)
    logger.info(f"Engine ready on {engine.device_name} (Provider: {engine.active_provider})")
    yield
    logger.info("Shutting down SER Engine...")

app = FastAPI(
    title="ViSEC SER Enterprise Engine",
    description="High-Performance Speech Emotion Recognition API (FastAPI + ONNX Runtime)",
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
    if engine is None:
        return HealthResponse(status="starting", device="unknown", active_provider="none", model_loaded=False)
    status_info = engine.get_status()
    return HealthResponse(
        status="online",
        device=status_info["device"],
        active_provider=status_info["active_provider"],
        model_loaded=True
    )

@app.post("/api/v1/predict/emotion", response_model=EmotionPredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    """
    Endpoint chính nhận diện cảm xúc từ file audio tải lên hoặc từ micro browser.
    """
    total_start = time.perf_counter()
    
    try:
        # 1. Đọc audio bytes bất đồng bộ
        try:
            audio_bytes = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Không thể đọc file âm thanh: {str(e)}")

        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="File âm thanh rỗng.")

        # 2. Tiền xử lý DSP (Decoded, VAD Silence Trim, RMS, Adaptive Chunking)
        dsp_start = time.perf_counter()
        try:
            dsp_result = preprocess_audio_pipeline(audio_bytes)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Lỗi xử lý file âm thanh: {str(e)}")
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

        # 3. Suy luận AI qua ONNX Runtime (Dynamic Batch Inference)
        infer_start = time.perf_counter()
        chunks = dsp_result["chunks"] # (K, 56000)
        chunk_probs = engine.predict_chunks(chunks) # (K, 4)
        infer_duration_ms = (time.perf_counter() - infer_start) * 1000.0

        # 4. Tổng hợp xác suất theo Salience Pooling
        final_probs = aggregate_salience_probabilities(chunk_probs)
        
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
