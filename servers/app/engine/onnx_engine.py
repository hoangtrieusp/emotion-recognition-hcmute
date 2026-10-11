import os
import time
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Tuple

logger = logging.getLogger("ser_engine")

EMOTION_MAPPING = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

class ONNXInferenceEngine:
    _instance = None

    def __init__(self, model_path: str = "servers/weights/ser_model.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found at path: {model_path}")

        # Tự động phát hiện Hardware Execution Providers
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device_name = "NVIDIA CUDA GPU"
        else:
            self.providers = ["CPUExecutionProvider"]
            self.device_name = "CPU"

        logger.info(f"Initializing ONNX Session with providers: {self.providers}")
        
        # Session options tối ưu hóa luồng
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=self.providers)
        self.active_provider = self.session.get_providers()[0]
        
        # Warm-up run để pre-allocate bộ nhớ, triệt tiêu độ trễ lượt gọi đầu tiên
        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((1, 16000 * 3), dtype=np.float32)
            _ = self.session.run(None, {"input_values": dummy})
            logger.info("ONNX Session warm-up completed successfully.")
        except Exception as e:
            logger.warning(f"ONNX Session warm-up warning: {e}")

    @classmethod
    def get_instance(cls, model_path: str = "servers/weights/ser_model.onnx"):
        if cls._instance is not None:
            return cls._instance
        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            logger.warning(
                f"[FALLBACK TRIGGERED] Không thể khởi tạo ONNX Engine ({e}). "
                "Tự động kích hoạt Fallback sang PyTorchEngine (model_fold_5.pt)..."
            )
            from servers.app.engine.pytorch_engine import PyTorchInferenceEngine
            cls._instance = PyTorchInferenceEngine.get_instance()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """
        Dự đoán phân phối xác suất cho batch K chunks.
        Input: chunks shape (K, window_size)
        Output: probabilities shape (K, 4)
        """
        ort_inputs = {"input_values": chunks}
        ort_outputs = self.session.run(None, ort_inputs)
        return ort_outputs[0] # Shape (K, 4)

    def get_status(self) -> Dict[str, str]:
        return {
            "device": self.device_name,
            "active_provider": self.active_provider,
            "providers": str(self.providers)
        }
