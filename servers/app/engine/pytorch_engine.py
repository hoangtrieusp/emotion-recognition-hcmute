import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2Model
from typing import Dict, Optional

logger = logging.getLogger("ser_pytorch_engine")

HF_MODEL_ID = "facebook/wav2vec2-base"

class ArcFaceInferenceHead(nn.Module):
    def __init__(self, num_classes=4, embedding_dim=256):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))

    def forward(self, embeddings):
        norm_emb = F.normalize(embeddings, p=2, dim=1)
        norm_w = F.normalize(self.weight, p=2, dim=1)
        return F.linear(norm_emb, norm_w)

class SERModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        config = Wav2Vec2Config.from_pretrained(HF_MODEL_ID)
        config.output_hidden_states = True
        self.wav2vec2 = Wav2Vec2Model(config)
        self.wav2vec2.gradient_checkpointing_disable()
        self.wav2vec2.freeze_feature_encoder()

        hidden_size = config.hidden_size
        self.layer_weights = nn.Parameter(torch.ones(13))
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feat_drop = nn.Dropout(0.3)

        self.bigru = nn.GRU(
            input_size=hidden_size, hidden_size=hidden_size // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self.att_proj = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.Tanh(), nn.Linear(128, 1)
        )

        self.sap_norm = nn.LayerNorm(hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, 256)
        self.gelu = nn.GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in [0.2, 0.3, 0.4, 0.5, 0.6]])

    def forward(self, input_values):
        outputs = self.wav2vec2(
            input_values=input_values,
            output_hidden_states=True,
            return_dict=True
        )
        stacked_states = torch.stack(outputs.hidden_states, dim=0)
        norm_weights = F.softmax(self.layer_weights, dim=0)
        weighted_feat = torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked_states, dim=0)
        weighted_feat = self.feat_drop(self.layer_norm(weighted_feat))

        gru_out, _ = self.bigru(weighted_feat)
        attn_scores = self.att_proj(gru_out).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        weighted_mean = torch.sum(attn_weights * gru_out, dim=1)
        diff = gru_out - weighted_mean.unsqueeze(1)
        weighted_var = torch.sum(attn_weights * (diff ** 2), dim=1)
        weighted_std = torch.sqrt(weighted_var + 1e-8)

        sap_out = torch.cat([weighted_mean, weighted_std], dim=-1)
        x = self.sap_norm(sap_out)
        x = self.gelu(self.fc1(x))
        return x

class EndToEndSER(nn.Module):
    def __init__(self, backbone: SERModel, head: ArcFaceInferenceHead, scale: float = 20.0):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.scale = scale

    def forward(self, input_values):
        embeddings = self.backbone(input_values)
        cosine = self.head(embeddings)
        logits = cosine * self.scale
        probs = torch.softmax(logits, dim=-1)
        return probs

def load_pytorch_model(checkpoint_path: str = "model_fold_5.pt", device: Optional[torch.device] = None) -> EndToEndSER:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PyTorch Checkpoint not found at: {checkpoint_path}")

    backbone = SERModel().to(device)
    head = ArcFaceInferenceHead(num_classes=4, embedding_dim=256).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["model_state"], strict=True)
    head.load_state_dict(checkpoint["criterion_state"], strict=False)

    model = EndToEndSER(backbone, head).to(device)
    model.eval()
    return model

class PyTorchInferenceEngine:
    _instance = None

    def __init__(self, checkpoint_path: str = "model_fold_5.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = f"{'NVIDIA CUDA GPU' if self.device.type == 'cuda' else 'CPU'} (PyTorch Fallback)"
        self.active_provider = "PyTorchFallbackExecutionProvider"
        self.providers = ["PyTorch"]
        
        logger.info(f"Khởi tạo PyTorchInferenceEngine trên thiết bị: {self.device}...")
        self.model = load_pytorch_model(checkpoint_path, self.device)
        logger.info("PyTorchInferenceEngine nạp checkpoint thành công.")

    @classmethod
    def get_instance(cls, checkpoint_path: str = "model_fold_5.pt") -> "PyTorchInferenceEngine":
        if cls._instance is None:
            cls._instance = cls(checkpoint_path)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        cls._instance = None

    def predict_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """
        Dự đoán phân phối xác suất cho batch K chunks bằng PyTorch.
        Input: chunks shape (K, window_size)
        Output: probabilities shape (K, 4)
        """
        with torch.no_grad():
            tensor = torch.from_numpy(chunks).float().to(self.device)
            probs = self.model(tensor)
            return probs.cpu().numpy()

    def get_status(self) -> Dict[str, str]:
        return {
            "device": self.device_name,
            "active_provider": self.active_provider,
            "providers": str(self.providers)
        }
