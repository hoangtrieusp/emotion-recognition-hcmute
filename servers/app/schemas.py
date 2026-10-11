from pydantic import BaseModel
from typing import Dict, Optional

class Telemetry(BaseModel):
    duration: float
    rms: float
    num_chunks: int
    dsp_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float

class EmotionPredictionResponse(BaseModel):
    status: str
    dominant_emotion: Optional[str] = None
    dominant_probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    telemetry: Optional[Telemetry] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    device: str
    active_provider: str
    model_loaded: bool
