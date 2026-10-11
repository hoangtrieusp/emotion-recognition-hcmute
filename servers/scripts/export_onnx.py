import os
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Config, Wav2Vec2Model

HF_MODEL_ID = "facebook/wav2vec2-base"
CHECKPOINT_PATH = "model_fold_5.pt"
OUTPUT_DIR = os.path.join("servers", "weights")
OUTPUT_ONNX_PATH = os.path.join(OUTPUT_DIR, "ser_model.onnx")

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
        # Trong inference mode: x chính là embedding (dropout là identity)
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

def export():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"[1/4] Khởi tạo mô hình và tải checkpoint '{CHECKPOINT_PATH}'...")
    
    backbone = SERModel().to(device)
    head = ArcFaceInferenceHead(num_classes=4, embedding_dim=256).to(device)
    
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Không tìm thấy file trọng số {CHECKPOINT_PATH}")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    backbone.load_state_dict(checkpoint["model_state"], strict=True)
    head.load_state_dict(checkpoint["criterion_state"], strict=False)
    
    e2e_model = EndToEndSER(backbone, head).to(device)
    e2e_model.eval()
    print("[SUCCESS] Đã nạp thành công trọng số vào EndToEndSERModel.")

    # Dummy input: 1 audio sample dài 3 giây ở 16kHz
    dummy_input = torch.randn(1, 16000 * 3, dtype=torch.float32, device=device)

    print(f"[2/4] Bắt đầu xuất ONNX sang '{OUTPUT_ONNX_PATH}'...")
    dynamic_axes = {
        "input_values": {0: "batch_size", 1: "sequence_length"},
        "probabilities": {0: "batch_size"}
    }

    with torch.no_grad():
        torch.onnx.export(
            e2e_model,
            dummy_input,
            OUTPUT_ONNX_PATH,
            input_names=["input_values"],
            output_names=["probabilities"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True
        )
    print(f"[SUCCESS] Đã xuất file ONNX thành công! Dung lượng: {os.path.getsize(OUTPUT_ONNX_PATH) / 1024 / 1024:.2f} MB")

    print("[3/4] Kiểm định sai số số học (Numerical Parity Check)...")
    import onnxruntime as ort

    ort_session = ort.InferenceSession(OUTPUT_ONNX_PATH, providers=["CPUExecutionProvider"])
    
    # Test trên 2 audio có độ dài khác nhau để kiểm tra tính linh hoạt của Dynamic Axes
    test_lengths = [16000 * 2, 16000 * 3.5]
    for idx, length in enumerate(test_lengths):
        test_input = torch.randn(1, int(length), dtype=torch.float32, device=device)
        
        with torch.no_grad():
            pt_out = e2e_model(test_input).numpy()
            
        ort_out = ort_session.run(None, {"input_values": test_input.numpy()})[0]
        
        mae = np.max(np.abs(pt_out - ort_out))
        print(f"Test case {idx+1} (Length {length/16000:.1f}s) - Max Absolute Error: {mae:.6e}")
        assert mae < 1e-4, f"Sai số quá lớn: {mae} >= 1e-4"

    print("[4/4] KIỂM ĐỊNH HOÀN TẤT: Mô hình ONNX đạt chuẩn Parity tuyệt đối ($MAE < 1e-4$)!")

if __name__ == "__main__":
    export()
