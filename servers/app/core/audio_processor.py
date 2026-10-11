import io
import math
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Dict, Any, Optional

TARGET_SR = 16000
MAX_CHUNK_DURATION = 3.5
HOP_DURATION = 2.0
WINDOW_SAMPLES = int(MAX_CHUNK_DURATION * TARGET_SR) # 56000
HOP_SAMPLES = int(HOP_DURATION * TARGET_SR)         # 32000

def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Giải mã audio bytes từ mọi định dạng (WAV, MP3, WebM/Opus, OGG, M4A, FLAC)
    về mảng NumPy Float32 16kHz Mono [-1.0, 1.0].
    """
    try:
        # Thử đọc nhanh bằng soundfile trước (rất nhanh với WAV, FLAC, OGG chuẩn)
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1) # Chuyển stereo -> mono
    except Exception:
        # Fallback qua pydub (hỗ trợ ffmpeg đọc WebM, M4A, MP3, etc.)
        segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        segment = segment.set_frame_rate(TARGET_SR).set_channels(1)
        raw_samples = np.array(segment.get_array_of_samples())
        
        # Chuẩn hóa về [-1.0, 1.0] dựa theo bit depth
        if segment.sample_width == 2:
            data = raw_samples.astype(np.float32) / 32768.0
        elif segment.sample_width == 4:
            data = raw_samples.astype(np.float32) / 2147483648.0
        else:
            data = raw_samples.astype(np.float32) / (2 ** (segment.sample_width * 8 - 1))
        sr = TARGET_SR

    # Resample về 16kHz nếu soundfile đọc ra sample rate khác
    if sr != TARGET_SR:
        duration = len(data) / sr
        num_target_samples = int(duration * TARGET_SR)
        data = np.interp(
            np.linspace(0, len(data), num_target_samples, endpoint=False),
            np.arange(len(data)),
            data
        ).astype(np.float32)
        sr = TARGET_SR

    return data, sr

def compute_rms(signal: np.ndarray) -> float:
    """Tính RMS Energy trung bình của tín hiệu âm thanh."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2) + 1e-9))

def trim_silence(signal: np.ndarray, top_db: float = 30.0, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    Cắt khoảng lặng tĩnh 2 đầu dựa trên ngưỡng năng lượng dB (tương tự librosa.effects.trim nhưng thuần NumPy).
    """
    if len(signal) <= frame_length:
        return signal

    # Tính RMS theo từng frame
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0])
    )
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    
    # Chuyển sang dB tương đối so với frame to nhất
    max_rms = np.max(frame_rms)
    if max_rms <= 1e-9:
        return signal
    
    db = 20 * np.log10(frame_rms / max_rms + 1e-9)
    active_indices = np.where(db > -top_db)[0]
    
    if len(active_indices) == 0:
        return signal
        
    start_sample = max(0, active_indices[0] * hop_length)
    end_sample = min(len(signal), (active_indices[-1] + 1) * hop_length + frame_length)
    return signal[start_sample:end_sample]

def adaptive_overlapping_chunking(signal: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Phân tách audio thành K chunks 3.5s gối đầu (Overlapping Sliding Window).
    Bao phủ 100% thời lượng, xóa điểm mù thời gian.
    Output: Ma trận NumPy shape (K, window_size) hoặc (1, len) nếu ngắn hơn 3.5s.
    """
    total_len = len(signal)
    if total_len <= WINDOW_SAMPLES:
        return signal[np.newaxis, :] # (1, total_len)

    chunks = []
    start = 0
    while start + WINDOW_SAMPLES <= total_len:
        chunks.append(signal[start : start + WINDOW_SAMPLES])
        start += HOP_SAMPLES

    # Nếu còn dư phần đuôi chưa hết, thêm chunk lùi từ đuôi
    if start < total_len:
        chunks.append(signal[-WINDOW_SAMPLES:])

    return np.stack(chunks, axis=0).astype(np.float32)

def aggregate_salience_probabilities(chunk_probs: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Confidence-Weighted Softmax Pooling:
    Tập hợp xác suất K chunks theo độ nổi bật cảm xúc (Salience),
    tránh việc các đoạn đều đều làm loãng cảm xúc giận dữ/vui bộc phát.
    """
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    # Salience = max_prob - min_prob cho từng chunk
    salience = np.max(chunk_probs, axis=1) - np.min(chunk_probs, axis=1)
    
    # Softmax trọng số theo temperature
    exp_weights = np.exp(salience / max(temperature, 1e-3))
    weights = exp_weights / np.sum(exp_weights)

    # Weighted sum qua các chunk
    final_probs = np.sum(weights[:, np.newaxis] * chunk_probs, axis=0)
    return final_probs.astype(np.float32)

def preprocess_audio_pipeline(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Toàn bộ luồng tiền xử lý DSP:
    Decode -> Resample -> Trim -> RMS check -> Chunking
    """
    raw_signal, sr = decode_audio_bytes(audio_bytes)
    
    # Cắt khoảng lặng 2 đầu (top_db=30)
    trimmed_signal = trim_silence(raw_signal, top_db=30.0)
    signal = trimmed_signal if len(trimmed_signal) > 0 else raw_signal
    
    duration = float(len(signal) / sr)
    rms_energy = compute_rms(signal)
    
    # Kiểm tra chất lượng tín hiệu
    if duration < 0.5:
        return {
            "is_valid": False,
            "error": "File âm thanh quá ngắn (< 0.5s). Vui lòng nói dài hơn.",
            "duration": duration,
            "rms": rms_energy
        }
    if rms_energy < 0.002:
        return {
            "is_valid": False,
            "error": "Không phát hiện tiếng người rõ ràng (Nhiễu nền hoặc im lặng).",
            "duration": duration,
            "rms": rms_energy
        }

    # Chuẩn hóa tín hiệu: Zero-mean, Unit-variance
    norm_signal = (signal - np.mean(signal)) / np.sqrt(np.var(signal) + 1e-7)
    
    # Phân chia chunks gối đầu
    chunks = adaptive_overlapping_chunking(norm_signal, sr=sr)
    
    return {
        "is_valid": True,
        "error": None,
        "chunks": chunks,
        "duration": duration,
        "rms": rms_energy,
        "num_chunks": chunks.shape[0]
    }
