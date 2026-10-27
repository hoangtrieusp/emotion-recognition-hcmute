from pydantic import BaseModel, Field
from typing import Dict, Optional

class Telemetry(BaseModel):
    duration: float = Field(..., description="Thời lượng âm thanh (giây)")
    rms: float = Field(..., description="Năng lượng RMS trung bình")
    num_chunks: int = Field(..., description="Số lượng chunk được chia tách")
    dsp_latency_ms: float = Field(..., description="Độ trễ xử lý tín hiệu số DSP (ms)")
    inference_latency_ms: float = Field(..., description="Độ trễ suy luận AI ONNX/PyTorch (ms)")
    total_latency_ms: float = Field(..., description="Tổng thời gian phản hồi E2E (ms)")

class EmotionPredictionResponse(BaseModel):
    status: str
    dominant_emotion: Optional[str] = None
    dominant_probability: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None
    telemetry: Optional[Telemetry] = None
    error: Optional[str] = None

class SystemMetrics(BaseModel):
    cpu_usage_percent: float = Field(..., description="Tỷ lệ sử dụng CPU (%)")
    ram_used_mb: float = Field(..., description="RAM đang sử dụng (MB)")
    ram_total_mb: float = Field(..., description="Tổng dung lượng RAM hệ thống (MB)")
    vram_used_mb: Optional[float] = Field(None, description="VRAM GPU đang sử dụng (MB)")
    vram_total_mb: Optional[float] = Field(None, description="Tổng dung lượng VRAM GPU (MB)")
    gpu_name: Optional[str] = Field(None, description="Tên card đồ họa GPU")

class HealthResponse(BaseModel):
    status: str
    device: str
    active_provider: str
    model_loaded: bool
    system_metrics: Optional[SystemMetrics] = None

