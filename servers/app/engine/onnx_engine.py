import os
import sys
import logging
from typing import Dict, Any, Optional, List
import numpy as np

# UTF-8 standard streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logger = logging.getLogger("ser_engine")

EMOTION_MAPPING: Dict[int, str] = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

def resolve_model_path(filename: str, preferred_dir: Optional[str] = None) -> str:
    """
    Tự động tìm kiếm đường dẫn tuyệt đối của file model/checkpoint
    dù server được khởi động từ thư mục gốc hay thư mục con.
    """
    candidate_paths: List[str] = []
    
    if preferred_dir:
        candidate_paths.append(os.path.join(preferred_dir, filename))
        candidate_paths.append(preferred_dir)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    servers_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    root_dir = os.path.abspath(os.path.join(servers_dir, ".."))

    candidate_paths.extend([
        os.path.join(servers_dir, "weights", filename),
        os.path.join(servers_dir, filename),
        os.path.join(root_dir, "servers", "weights", filename),
        os.path.join(root_dir, filename),
        os.path.join(root_dir, "MODELs", "model", filename),
        os.path.join(os.getcwd(), "servers", "weights", filename),
        os.path.join(os.getcwd(), "weights", filename),
        os.path.join(os.getcwd(), filename),
    ])

    for path in candidate_paths:
        if os.path.exists(path) and os.path.isfile(path):
            return os.path.abspath(path)

    return os.path.abspath(os.path.join(servers_dir, "weights", filename))


class PyTorchFallbackEngine:
    """
    Disaster Recovery Engine: Sử dụng PyTorch checkpoint trực tiếp (model_fold_5.pt)
    khi mô hình ONNX bị lỗi, hỏng file hoặc thiếu môi trường ONNX Runtime.
    Tự động tận dụng NVIDIA CUDA GPU nếu khả dụng, ngược lại dùng CPU.
    """
    _instance: Optional["PyTorchFallbackEngine"] = None

    def __init__(self, checkpoint_path: Optional[str] = None):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.torch = torch
        resolved_ckpt = resolve_model_path(checkpoint_path or "model_fold_5.pt")
        
        if not os.path.exists(resolved_ckpt):
            raise FileNotFoundError(f"[Disaster Recovery] Không tìm thấy PyTorch checkpoint tại: {resolved_ckpt}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = f"{'NVIDIA CUDA GPU' if self.device.type == 'cuda' else 'CPU'} (PyTorch Fallback)"
        self.active_provider = "PyTorchFallbackExecutionProvider"
        self.providers = [f"PyTorch-{self.device.type.upper()}"]

        logger.info(f"[Fallback Engine] Khởi tạo PyTorch trên thiết bị: {self.device_name}")

        class ArcFaceInferenceHead(nn.Module):
            def __init__(self, num_classes=4, embedding_dim=256):
                super().__init__()
                self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))

            def forward(self, embeddings):
                norm_emb = F.normalize(embeddings, p=2, dim=1)
                norm_w = F.normalize(self.weight, p=2, dim=1)
                return F.linear(norm_emb, norm_w)

        class SERModel(nn.Module):
            def __init__(self):
                super().__init__()
                cfg = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
                cfg.output_hidden_states = True
                self.wav2vec2 = Wav2Vec2Model(cfg)
                self.wav2vec2.freeze_feature_encoder()
                h = cfg.hidden_size
                self.layer_weights = nn.Parameter(torch.ones(13))
                self.layer_norm = nn.LayerNorm(h)
                self.feat_drop = nn.Dropout(0.3)
                self.bigru = nn.GRU(h, h // 2, batch_first=True, bidirectional=True)
                self.att_proj = nn.Sequential(nn.Linear(h, 128), nn.Tanh(), nn.Linear(128, 1))
                self.sap_norm = nn.LayerNorm(h * 2)
                self.fc1 = nn.Linear(h * 2, 256)
                self.gelu = nn.GELU()

            def forward(self, input_values):
                out = self.wav2vec2(input_values=input_values, output_hidden_states=True, return_dict=True)
                norm_w = F.softmax(self.layer_weights, dim=0)
                feat = self.feat_drop(self.layer_norm(torch.sum(norm_w.view(-1, 1, 1, 1) * torch.stack(out.hidden_states), dim=0)))
                gru_out, _ = self.bigru(feat)
                attn = F.softmax(self.att_proj(gru_out).squeeze(-1), dim=-1).unsqueeze(-1)
                mean = torch.sum(attn * gru_out, dim=1)
                std = torch.sqrt(torch.sum(attn * ((gru_out - mean.unsqueeze(1)) ** 2), dim=1) + 1e-8)
                return self.gelu(self.fc1(self.sap_norm(torch.cat([mean, std], dim=-1))))

        class EndToEnd(nn.Module):
            def __init__(self, backbone, head, scale=20.0):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.scale = scale

            def forward(self, x):
                return torch.softmax(self.head(self.backbone(x)) * self.scale, dim=-1)

        backbone = SERModel().to(self.device)
        head = ArcFaceInferenceHead().to(self.device)
        ckpt = torch.load(resolved_ckpt, map_location=self.device, weights_only=False)
        backbone.load_state_dict(ckpt["model_state"], strict=True)
        head.load_state_dict(ckpt["criterion_state"], strict=False)
        self.model = EndToEnd(backbone, head).to(self.device)
        self.model.eval()

        logger.info(f"[Fallback Engine] Nạp trọng số PyTorch từ: {resolved_ckpt}")

    @classmethod
    def get_instance(cls, checkpoint_path: Optional[str] = None) -> "PyTorchFallbackEngine":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.from_numpy(chunks).float().to(self.device)
            return self.model(t).cpu().numpy()

    def get_status(self) -> Dict[str, str]:
        return {
            "device": self.device_name,
            "active_provider": self.active_provider,
            "providers": str(self.providers),
        }


class ONNXInferenceEngine:
    """
    Singleton AI Engine:
    - Resilient Dual-Engine: CUDAExecutionProvider -> CPUExecutionProvider -> PyTorch Disaster Recovery
    - Khởi tạo duy nhất 1 lần trong Lifespan event
    """
    _instance: Optional[Any] = None

    def __init__(self, model_path: Optional[str] = None):
        import onnxruntime as ort

        resolved_onnx = resolve_model_path(model_path or "ser_model.onnx")
        if not os.path.exists(resolved_onnx):
            raise FileNotFoundError(f"Không tìm thấy file mô hình ONNX tại: {resolved_onnx}")

        self.model_path = resolved_onnx
        available_providers = ort.get_available_providers()

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

        session_created = False
        last_error = None

        # Tier 1: CUDAExecutionProvider nếu có phần cứng NVIDIA
        if "CUDAExecutionProvider" in available_providers:
            try:
                tier1_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=tier1_providers)
                self.providers = self.session.get_providers()
                self.active_provider = self.providers[0]
                self.device_name = "NVIDIA CUDA GPU" if "CUDA" in self.active_provider else "CPU"
                self._warmup()
                session_created = True
                logger.info(f"ONNX kích hoạt trên: {self.device_name} ({self.active_provider})")
            except Exception as e:
                logger.warning(f"CUDA provider khởi tạo không thành công ({e}), chuyển sang CPU provider.")
                last_error = e

        # Tier 2: Fallback sang CPUExecutionProvider
        if not session_created:
            try:
                tier2_providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=tier2_providers)
                self.providers = self.session.get_providers()
                self.active_provider = self.providers[0]
                self.device_name = "CPU"
                self._warmup()
                session_created = True
                logger.info(f"ONNX kích hoạt trên CPU ({self.active_provider})")
            except Exception as e:
                last_error = e
                raise last_error

    def _warmup(self) -> None:
        """Warmup đồ thị ONNX Runtime để triệt tiêu trễ ở request đầu tiên."""
        try:
            dummy = np.zeros((1, 16000 * 3), dtype=np.float32)
            _ = self.session.run(None, {"input_values": dummy})
        except Exception as e:
            logger.warning(f"Cảnh báo Warmup ONNX: {e}")

    @classmethod
    def get_instance(cls, model_path: Optional[str] = None) -> Any:
        """Singleton Accessor với cơ chế tự động Fallback sang PyTorch."""
        if cls._instance is not None:
            return cls._instance

        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            logger.warning(f"Khởi tạo ONNX Engine không thành công ({e}), kích hoạt PyTorch Fallback Engine.")
            cls._instance = PyTorchFallbackEngine.get_instance()
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None
        PyTorchFallbackEngine.reset_instance()

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input_values": chunks})[0]

    def get_status(self) -> Dict[str, str]:
        return {
            "device": self.device_name,
            "active_provider": self.active_provider,
            "providers": str(self.providers),
        }
