import os
import logging
import numpy as np
import onnxruntime as ort
from typing import Dict, Any

logger = logging.getLogger("ser_engine")

EMOTION_MAPPING = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}

class PyTorchFallbackEngine:
    """Fallback engine using raw PyTorch checkpoint if ONNX model is missing or corrupt."""
    _instance = None

    def __init__(self, checkpoint_path: str = "model_fold_5.pt"):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from transformers import Wav2Vec2Config, Wav2Vec2Model

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = f"{'NVIDIA CUDA GPU' if self.device.type == 'cuda' else 'CPU'} (PyTorch Fallback)"
        self.active_provider = "PyTorchFallbackExecutionProvider"
        self.providers = ["PyTorch"]

        class ArcFaceInferenceHead(nn.Module):
            def __init__(self, num_classes=4, embedding_dim=256):
                super().__init__()
                self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
            def forward(self, embeddings):
                return F.linear(F.normalize(embeddings, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

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
                self.backbone, self.head, self.scale = backbone, head, scale
            def forward(self, x):
                return torch.softmax(self.head(self.backbone(x)) * self.scale, dim=-1)

        backbone = SERModel().to(self.device)
        head = ArcFaceInferenceHead().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        backbone.load_state_dict(ckpt["model_state"], strict=True)
        head.load_state_dict(ckpt["criterion_state"], strict=False)
        self.model = EndToEnd(backbone, head).to(self.device)
        self.model.eval()

    @classmethod
    def get_instance(cls, checkpoint_path: str = "model_fold_5.pt"):
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            t = self.torch.from_numpy(chunks).float().to(self.device)
            return self.model(t).cpu().numpy()

    def get_status(self) -> Dict[str, str]:
        return {"device": self.device_name, "active_provider": self.active_provider, "providers": str(self.providers)}


class ONNXInferenceEngine:
    _instance = None

    def __init__(self, model_path: str = "servers/weights/ser_model.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX Model not found at path: {model_path}")

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device_name = "NVIDIA CUDA GPU"
        else:
            self.providers = ["CPUExecutionProvider"]
            self.device_name = "CPU"

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=self.providers)
        self.active_provider = self.session.get_providers()[0]
        self._warmup()

    def _warmup(self):
        try:
            dummy = np.zeros((1, 16000 * 3), dtype=np.float32)
            _ = self.session.run(None, {"input_values": dummy})
        except Exception as e:
            logger.warning(f"ONNX Warmup warning: {e}")

    @classmethod
    def get_instance(cls, model_path: str = "servers/weights/ser_model.onnx"):
        if cls._instance is not None:
            return cls._instance
        try:
            cls._instance = cls(model_path)
            return cls._instance
        except Exception as e:
            logger.warning(f"[FALLBACK TRIGGERED] ONNX initialization failed ({e}). Reverting to PyTorch engine...")
            cls._instance = PyTorchFallbackEngine.get_instance()
            return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None
        PyTorchFallbackEngine.reset_instance()

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input_values": chunks})[0]

    def get_status(self) -> Dict[str, str]:
        return {"device": self.device_name, "active_provider": self.active_provider, "providers": str(self.providers)}
