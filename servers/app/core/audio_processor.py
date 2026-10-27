import io
import math
import sys
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from typing import Tuple, Dict, Any, Optional

# Tự động nạp độc lập standalone FFmpeg từ imageio-ffmpeg nếu máy người dùng chưa cài FFmpeg toàn cục
try:
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_bin
    AudioSegment.ffmpeg = ffmpeg_bin
except Exception:
    pass

# UTF-8 standard streams on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_SR: int = 16000
MAX_CHUNK_DURATION: float = 3.5
HOP_DURATION: float = 2.0
WINDOW_SAMPLES: int = int(MAX_CHUNK_DURATION * TARGET_SR)  # 56,000 samples (3.5s)
HOP_SAMPLES: int = int(HOP_DURATION * TARGET_SR)          # 32,000 samples (2.0s)

def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Giải mã audio bytes từ mọi định dạng (WAV, MP3, WebM/Opus, OGG, M4A, FLAC)
    về mảng NumPy Float32 16kHz Mono [-1.0, 1.0] hoàn toàn trong RAM (Zero-Disk I/O).
    """
    data: Optional[np.ndarray] = None
    sr: int = TARGET_SR

    try:
        # 1. Đọc trực tiếp bằng SoundFile (tối ưu cho WAV, FLAC, OGG chuẩn)
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Chuyển stereo -> mono
    except Exception:
        # 2. Fallback qua PyDub/FFmpeg in-memory (hỗ trợ WebM/Opus từ trình duyệt, MP3, M4A)
        try:
            with io.BytesIO(audio_bytes) as bio:
                segment = AudioSegment.from_file(bio)
                segment = segment.set_frame_rate(TARGET_SR).set_channels(1)
                raw_samples = np.array(segment.get_array_of_samples())
                
                if len(raw_samples) == 0:
                    raise ValueError("File âm thanh không chứa mẫu tín hiệu hợp lệ.")

                # Chuẩn hóa về [-1.0, 1.0] theo bit depth
                if segment.sample_width == 2:
                    data = raw_samples.astype(np.float32) / 32768.0
                elif segment.sample_width == 4:
                    data = raw_samples.astype(np.float32) / 2147483648.0
                else:
                    data = raw_samples.astype(np.float32) / float(2 ** (segment.sample_width * 8 - 1))
                sr = TARGET_SR
                del segment
                del raw_samples
        except Exception as e:
            err_msg = str(e)
            if "WinError 2" in err_msg or "ffmpeg" in err_msg.lower():
                raise ValueError("Máy chủ chưa cài đặt công cụ FFmpeg để giải mã âm thanh ghi âm từ trình duyệt. Vui lòng mở terminal chạy lệnh: pip install imageio-ffmpeg")
            raise ValueError(f"Không thể giải mã dữ liệu âm thanh: {err_msg}")

    if data is None or len(data) == 0:
        raise ValueError("Tín hiệu âm thanh sau khi giải mã rỗng.")

    # 3. Resample về 16kHz nếu cần
    if sr != TARGET_SR:
        duration = len(data) / float(sr)
        num_target_samples = int(duration * TARGET_SR)
        data = np.interp(
            np.linspace(0, len(data), num_target_samples, endpoint=False),
            np.arange(len(data)),
            data
        ).astype(np.float32)
        sr = TARGET_SR

    return data, sr

def compute_rms(signal: np.ndarray) -> float:
    """Tính năng lượng hiệu dụng RMS Energy trung bình của tín hiệu."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal ** 2) + 1e-9))

def trim_silence(signal: np.ndarray, top_db: float = 30.0, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    Cắt khoảng lặng tĩnh 2 đầu (VAD) dựa trên ngưỡng năng lượng dB (thuần NumPy).
    """
    if len(signal) <= frame_length:
        return signal

    # Tính RMS trượt với Stride Tricks
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0])
    )
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
    
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
    Phân tách audio thành K chunks 3.5s gối đầu (hop 2.0s), bao phủ 100% thời lượng.
    Output: Ma trận NumPy shape (K, 56000) hoặc (1, len) nếu ngắn hơn 3.5s.
    """
    total_len = len(signal)
    if total_len <= WINDOW_SAMPLES:
        return signal[np.newaxis, :]

    chunks = []
    start = 0
    while start + WINDOW_SAMPLES <= total_len:
        chunks.append(signal[start : start + WINDOW_SAMPLES])
        start += HOP_SAMPLES

    # Bổ sung chunk vét đuôi lùi từ cuối file
    if start < total_len:
        chunks.append(signal[-WINDOW_SAMPLES:])

    return np.stack(chunks, axis=0).astype(np.float32)

def aggregate_salience_probabilities(chunk_probs: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """
    Confidence-Weighted Salience Pooling:
    Tập hợp xác suất K chunks theo độ nổi bật cảm xúc (Salience = max - min),
    bảo toàn các biểu cảm bộc phát ngắn hạn (quát giận, cười rộ).
    """
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    salience = np.max(chunk_probs, axis=1) - np.min(chunk_probs, axis=1)
    exp_weights = np.exp(salience / max(temperature, 1e-3))
    weights = exp_weights / np.sum(exp_weights)

    final_probs = np.sum(weights[:, np.newaxis] * chunk_probs, axis=0)
    return final_probs.astype(np.float32)

def preprocess_audio_pipeline(audio_bytes: bytes) -> Dict[str, Any]:
    """
    Pipeline tiền xử lý DSP hoàn chỉnh:
    Decode -> Resample -> VAD Silence Trim -> RMS Check -> Normalize -> Adaptive Chunking
    """
    raw_signal, sr = decode_audio_bytes(audio_bytes)
    
    # Cắt khoảng lặng 2 đầu (top_db=30.0)
    trimmed_signal = trim_silence(raw_signal, top_db=30.0)
    signal = trimmed_signal if len(trimmed_signal) > 0 else raw_signal
    
    duration = float(len(signal) / sr)
    rms_energy = compute_rms(signal)
    
    # Kiểm tra chất lượng tín hiệu (Quality Gating)
    if duration < 0.5:
        return {
            "is_valid": False,
            "error": "File âm thanh quá ngắn (< 0.5s). Vui lòng cung cấp đoạn thu âm dài hơn.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "num_chunks": 0
        }
    if rms_energy < 0.002:
        return {
            "is_valid": False,
            "error": "Không phát hiện tiếng người rõ ràng (Nhiễu nền hoặc im lặng, RMS < 0.002).",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "num_chunks": 0
        }

    # Chuẩn hóa Z-Score: Zero-mean, Unit-variance
    norm_signal = (signal - np.mean(signal)) / np.sqrt(np.var(signal) + 1e-7)
    
    # Phân đoạn cửa sổ trượt gối đầu
    chunks = adaptive_overlapping_chunking(norm_signal, sr=sr)
    
    return {
        "is_valid": True,
        "error": None,
        "chunks": chunks,
        "duration": duration,
        "rms": rms_energy,
        "num_chunks": chunks.shape[0]
    }
