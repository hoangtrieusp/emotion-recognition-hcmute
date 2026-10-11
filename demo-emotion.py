import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import gradio as gr
from transformers import Wav2Vec2Config, Wav2Vec2Model, AutoFeatureExtractor

# Tắt các cảnh báo không cần thiết từ HuggingFace để Console sạch sẽ
warnings.filterwarnings("ignore")

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & ĐỊNH TUYẾN
# ==========================================
HF_MODEL_ID = "facebook/wav2vec2-base"
EMOTION_MAPPING = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
COMPUTATION_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_FILEPATH = "model_fold_5.pt"

# ==========================================
# 2. KIẾN TRÚC MÔ HÌNH (STRICT REPRODUCTION)
# ==========================================
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

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(
            input_values=input_values, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True
        )
        stacked_states = torch.stack(outputs.hidden_states, dim=0)
        num_states = stacked_states.shape[0]

        if self.layer_weights.shape[0] != num_states:
            active_weights = self.layer_weights[:num_states] if self.layer_weights.shape[0] > num_states else F.pad(self.layer_weights, (0, num_states - self.layer_weights.shape[0]))
        else:
            active_weights = self.layer_weights

        norm_weights = F.softmax(active_weights, dim=0)
        weighted_feat = torch.sum(norm_weights.view(-1, 1, 1, 1) * stacked_states, dim=0)
        weighted_feat = self.feat_drop(self.layer_norm(weighted_feat))

        if attention_mask is not None:
            input_lens = attention_mask.sum(dim=-1)
            feature_lens = self.wav2vec2._get_feat_extract_output_lengths(input_lens)
            batch_size, seq_len, _ = weighted_feat.shape
            mask = torch.arange(seq_len, device=weighted_feat.device).expand(batch_size, seq_len) < feature_lens.unsqueeze(1)
            weighted_feat = weighted_feat * mask.unsqueeze(-1).float()
        else:
            mask = None

        gru_out, _ = self.bigru(weighted_feat)
        attn_scores = self.att_proj(gru_out).squeeze(-1)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        weighted_mean = torch.sum(attn_weights * gru_out, dim=1)
        diff = gru_out - weighted_mean.unsqueeze(1)
        weighted_var = torch.sum(attn_weights * (diff ** 2), dim=1)
        weighted_std = torch.sqrt(weighted_var + 1e-8)

        sap_out = torch.cat([weighted_mean, weighted_std], dim=-1)
        x = self.sap_norm(sap_out)
        x = self.gelu(self.fc1(x))

        embeddings = torch.mean(torch.stack([dropout(x) for dropout in self.dropouts], dim=0), dim=0)
        return embeddings

# ==========================================
# 3. KHỞI TẠO & NẠP TRỌNG SỐ (L0/L1 LAYER)
# ==========================================
try:
    inference_model = SERModel().to(COMPUTATION_DEVICE)
    classifier_head = ArcFaceInferenceHead(num_classes=4, embedding_dim=256).to(COMPUTATION_DEVICE)
    feature_extractor = AutoFeatureExtractor.from_pretrained(HF_MODEL_ID, do_normalize=False)

    if os.path.exists(CHECKPOINT_FILEPATH):
        checkpoint = torch.load(CHECKPOINT_FILEPATH, map_location=COMPUTATION_DEVICE, weights_only=False)
        inference_model.load_state_dict(checkpoint["model_state"], strict=True)
        classifier_head.load_state_dict(checkpoint["criterion_state"], strict=False)
        inference_model.eval()
        classifier_head.eval()
        print(f"[SUCCESS] Tải trọng số thành công từ '{CHECKPOINT_FILEPATH}' lên {COMPUTATION_DEVICE}")
    else:
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {CHECKPOINT_FILEPATH}")
except Exception as e:
    print(f"[FATAL ERROR] Lỗi khởi tạo mô hình: {e}")
    exit(1)

# ==========================================
# 4. ENGINE XỬ LÝ ÂM THANH (L2/L3 LAYER)
# ==========================================
def process_audio_chunk(signal, sr=16000):
    """Xử lý từng chunk âm thanh độc lập để lấy phân bố xác suất"""
    normalized_signal = (signal - np.mean(signal)) / np.sqrt(np.var(signal) + 1e-7)
    inputs = feature_extractor(normalized_signal, sampling_rate=sr, return_tensors="pt", padding=True)
    tensor_inputs = inputs.input_values.to(COMPUTATION_DEVICE)
    tensor_mask = inputs.attention_mask.to(COMPUTATION_DEVICE) if "attention_mask" in inputs else None

    with torch.inference_mode():
        with torch.amp.autocast(device_type=COMPUTATION_DEVICE.type):
            embeddings = inference_model(tensor_inputs, tensor_mask)
            cosine = classifier_head(embeddings)
            logits = cosine * 20.0
            probabilities = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return probabilities

def execute_inference(audio_filepath):
    """Pipeline Inference tích hợp Chunking & Voting"""
    if audio_filepath is None:
        return {"Lỗi: Chưa cung cấp âm thanh": 1.0}, "Không có dữ liệu"

    try:
        # Tự động đọc mọi định dạng (MP3, WAV, FLAC, M4A...) và ép về 16kHz Mono
        raw_signal, sr = librosa.load(audio_filepath, sr=16000, mono=True)

        # Tiền xử lý: Cắt khoảng lặng tĩnh ở 2 đầu (Threshold 40dB thay vì 60dB để nhạy hơn với mic ngoài)
        trimmed_signal, _ = librosa.effects.trim(raw_signal, top_db=30)
        signal = trimmed_signal if len(trimmed_signal) > 0 else raw_signal

        duration = len(signal) / sr
        rms_energy = np.mean(librosa.feature.rms(y=signal)[0])

        if duration < 0.5:
            return {"Lỗi: File quá ngắn (<0.5s)": 1.0}, f"Thời lượng: {duration:.2f}s | Lỗi"
        if rms_energy < 0.002:
            return {"Lỗi: Không phát hiện tiếng người (Nhiễu/Im lặng)": 1.0}, f"RMS: {rms_energy:.5f} | Trống"

        # Kỹ thuật Ensemble: Nếu file dài > 3.5s, cắt thành 2 chunk để vote tránh loãng tín hiệu
        MAX_LEN = int(3.5 * sr)
        if len(signal) > MAX_LEN:
            chunk1 = signal[:MAX_LEN]
            chunk2 = signal[-MAX_LEN:]
            prob1 = process_audio_chunk(chunk1)
            prob2 = process_audio_chunk(chunk2)
            final_probs = (prob1 + prob2) / 2.0  # Mean Pooling Probabilities
        else:
            final_probs = process_audio_chunk(signal)

        results = {EMOTION_MAPPING[idx]: float(final_probs[idx]) for idx in range(4)}
        metadata = f"Thời lượng: {duration:.2f}s | Tần số lấy mẫu: {sr}Hz | RMS: {rms_energy:.4f}"

        return results, metadata

    except Exception as e:
        return {f"Lỗi hệ thống: {str(e)}": 1.0}, "Xử lý thất bại"

# ==========================================
# 5. GIAO DIỆN UI/UX ĐẠT CHUẨN (L5 LAYER)
# ==========================================
custom_css = """
#component-0 { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.gradio-container { background-color: #f8f9fa; }
"""

with gr.Blocks(title="ViSEC SER Enterprise Pipeline", css=custom_css) as demo_interface:
    gr.Markdown("""
    # 🎙️ HỆ THỐNG NHẬN DIỆN CẢM XÚC TIẾNG NÓI (SER)
    **Pipeline:** Wav2Vec2 + BiGRU + Self-Attention Pooling + ArcFace Loss
    *Hỗ trợ đa định dạng (WAV, MP3, M4A, FLAC). Tự động lọc nhiễu môi trường.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # Tích hợp cả Mic và Tải file
            audio_input = gr.Audio(type="filepath", sources=["microphone", "upload"], label="Đầu vào Âm thanh (Mic/Upload)")
            analyze_btn = gr.Button("Phân Tích Cảm Xúc", variant="primary")

        with gr.Column(scale=1):
            # Output 1: Bar Chart (Label)
            emotion_output = gr.Label(num_top_classes=4, label="Phân phối Xác suất (Confidence Score)")
            # Output 2: Textbox cho Metadata âm học
            metadata_output = gr.Textbox(label="Phân tích Đặc trưng Âm học (Metadata)", interactive=False)

    analyze_btn.click(
        fn=execute_inference,
        inputs=audio_input,
        outputs=[emotion_output, metadata_output]
    )

if __name__ == "__main__":
    demo_interface.launch(share=False, debug=True)
