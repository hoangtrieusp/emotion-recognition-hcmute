import io
import math
import sys
import numpy as np
import soundfile as sf
import scipy.signal as signal
import scipy.ndimage as ndimage
from pydub import AudioSegment
from typing import Tuple, Dict, Any, Optional

try:
    import importlib
    _imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    _ffmpeg_bin = getattr(_imageio_ffmpeg, "get_ffmpeg_exe", lambda: None)()
    if _ffmpeg_bin:
        AudioSegment.converter = _ffmpeg_bin
        AudioSegment.ffmpeg = _ffmpeg_bin
except Exception:
    pass

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

TARGET_SR: int = 16000
MAX_CHUNK_DURATION: float = 3.5
HOP_DURATION: float = 2.0
WINDOW_SAMPLES: int = int(MAX_CHUNK_DURATION * TARGET_SR)
HOP_SAMPLES: int = int(HOP_DURATION * TARGET_SR)

def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    data: Optional[np.ndarray] = None
    sr: int = TARGET_SR

    try:
        with io.BytesIO(audio_bytes) as bio:
            data, sr = sf.read(bio, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
    except Exception:
        try:
            with io.BytesIO(audio_bytes) as bio:
                segment = AudioSegment.from_file(bio)
                segment = segment.set_frame_rate(TARGET_SR).set_channels(1)
                raw_samples = np.array(segment.get_array_of_samples())

                if len(raw_samples) == 0:
                    raise ValueError("File âm thanh không chứa mẫu tín hiệu hợp lệ.")

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

def compute_rms(signal_data: np.ndarray) -> float:
    if len(signal_data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(signal_data ** 2) + 1e-9))

def apply_bandpass_filter(signal_data: np.ndarray, sr: int = TARGET_SR, lowcut: float = 80.0, highcut: float = 7500.0) -> np.ndarray:
    if len(signal_data) < 512:
        return signal_data.astype(np.float32)
    sos = signal.butter(5, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, signal_data).astype(np.float32)

def compute_voicing_ratio(signal_data: np.ndarray, sr: int = TARGET_SR) -> float:
    frame_len = int(0.030 * sr)
    hop_len = int(0.010 * sr)
    num_f = (len(signal_data) - frame_len) // hop_len
    if num_f <= 0:
        return 0.0
    min_lag = int(sr / 450)
    max_lag = int(sr / 80)
    voiced_count = 0
    stride = 2
    for i in range(0, num_f, stride):
        fr = signal_data[i * hop_len : i * hop_len + frame_len]
        if np.sqrt(np.mean(fr ** 2) + 1e-9) > 0.001:
            ac = signal.correlate(fr, fr, mode="full")[frame_len - 1 :]
            r0 = ac[0] + 1e-9
            pitch_ac = ac[min_lag:max_lag]
            if len(pitch_ac) > 0 and (np.max(pitch_ac) / r0) >= 0.35:
                voiced_count += 1
    return float((voiced_count * stride) / num_f)

def apply_adaptive_spectral_cleanup(signal_data: np.ndarray, sr: int = TARGET_SR) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    if len(signal_data) < 512:
        return signal_data.astype(np.float32), np.ones(1, dtype=np.float32), 0.0, 30.0, 0.5, 0.5

    filtered = apply_bandpass_filter(signal_data, sr=sr)

    n_fft = 512
    hop = 256
    f, t, Zxx = signal.stft(filtered, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Zxx).astype(np.float32)
    phase = np.angle(Zxx)

    harm = ndimage.median_filter(mag, size=(1, 15))
    perc = ndimage.median_filter(mag, size=(15, 1))

    harm_energy = np.mean(harm ** 2)
    perc_energy = np.mean(perc ** 2)
    perc_ratio = float(perc_energy / (harm_energy + 1e-8))

    if perc_ratio > 0.5:
        adaptive_margin = float(np.clip(1.0 + 3.0 * (perc_ratio - 0.5), 1.0, 4.0))
        harm_blend = float(np.clip(0.40 + 1.2 * (perc_ratio - 0.5), 0.40, 0.98))
    else:
        adaptive_margin = 1.0
        harm_blend = 0.25

    mask_harm = (harm ** 2) / (harm ** 2 + (perc * adaptive_margin) ** 2 + 1e-8)

    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
    frame_rms = np.sqrt(np.mean(mag ** 2, axis=0))
    p90 = np.percentile(frame_rms, 90)
    p10 = np.percentile(frame_rms, 10)
    snr_db = float(20.0 * np.log10(p90 / (p10 + 1e-6) + 1e-4))

    wiener_strength = float(np.clip(1.3 - 0.03 * (snr_db - 10.0), 0.3, 1.5))
    snr_local = mag / (noise_floor + 1e-8)
    wiener_gain = np.clip(1.0 - wiener_strength / (snr_local + 1e-4), 0.12, 1.0)
    wiener_gain = ndimage.uniform_filter1d(wiener_gain, size=3, axis=1)

    final_gain = (harm_blend * mask_harm + (1.0 - harm_blend)) * wiener_gain

    cleaned_mag = mag * final_gain
    _, cleaned = signal.istft(cleaned_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    cleaned = cleaned[:len(signal_data)].astype(np.float32)

    total_energy_sum = float(np.sum(cleaned_mag ** 2) + 1e-8)
    f_fund = (f >= 85) & (f <= 450)
    fund_ratio = float(np.sum(cleaned_mag[f_fund, :] ** 2)) / total_energy_sum

    formant_bins = (f >= 300) & (f <= 3400)
    formant_energy = np.sum(cleaned_mag[formant_bins, :] ** 2, axis=0)
    total_energy = np.sum(cleaned_mag ** 2, axis=0) + 1e-8
    frame_speech = np.clip(formant_energy / total_energy, 0.05, 1.0)
    formant_ratio = float(np.sum(formant_energy)) / total_energy_sum

    return cleaned, frame_speech, perc_ratio, snr_db, fund_ratio, formant_ratio

def trim_silence_vad(signal_data: np.ndarray, frame_speech: np.ndarray, hop_length: int = 256, speech_threshold: float = 0.30) -> np.ndarray:
    if len(signal_data) <= WINDOW_SAMPLES:
        return signal_data

    active_indices = np.where(frame_speech > speech_threshold)[0]
    if len(active_indices) == 0:
        return signal_data

    padding_frames = int(0.2 * TARGET_SR / hop_length)
    start_frame = max(0, active_indices[0] - padding_frames)
    end_frame = min(len(frame_speech), active_indices[-1] + padding_frames + 1)

    start_sample = max(0, start_frame * hop_length)
    end_sample = min(len(signal_data), end_frame * hop_length)

    trimmed = signal_data[start_sample:end_sample]
    if len(trimmed) < int(0.5 * TARGET_SR):
        return signal_data
    return trimmed

def adaptive_overlapping_chunking(signal_data: np.ndarray, frame_speech: np.ndarray, sr: int = TARGET_SR) -> Tuple[np.ndarray, np.ndarray]:
    total_len = len(signal_data)
    hop = 256
    chunks = []
    weights = []

    if total_len <= WINDOW_SAMPLES:
        norm = (signal_data - np.mean(signal_data)) / np.sqrt(np.var(signal_data) + 1e-7)
        return norm[np.newaxis, :].astype(np.float32), np.array([1.0], dtype=np.float32)

    frame_len = 512
    num_f = total_len // frame_len
    if num_f > 0:
        f_rms_vals = np.array([np.sqrt(np.mean(signal_data[i * frame_len : (i + 1) * frame_len] ** 2)) for i in range(num_f)])
        voiced_f_rms = f_rms_vals[f_rms_vals > 0.003]
        primary_rms = float(np.percentile(voiced_f_rms, 80)) if len(voiced_f_rms) > 0 else 0.03
    else:
        primary_rms = 0.03

    start = 0
    while start + WINDOW_SAMPLES <= total_len:
        c = signal_data[start : start + WINDOW_SAMPLES]
        norm = (c - np.mean(c)) / np.sqrt(np.var(c) + 1e-7)
        chunks.append(norm)

        f_start = start // hop
        f_end = (start + WINDOW_SAMPLES) // hop
        speech_val = float(np.mean(frame_speech[f_start:f_end])) if f_end > f_start else 0.5
        c_rms = np.sqrt(np.mean(c ** 2) + 1e-9)
        rel_energy = float(np.clip((c_rms - 0.20 * primary_rms) / (0.40 * primary_rms + 1e-6), 0.02, 1.0))
        weights.append(speech_val * rel_energy)
        start += HOP_SAMPLES

    if start < total_len:
        c = signal_data[-WINDOW_SAMPLES:]
        norm = (c - np.mean(c)) / np.sqrt(np.var(c) + 1e-7)
        chunks.append(norm)
        f_start = (total_len - WINDOW_SAMPLES) // hop
        f_end = total_len // hop
        speech_val = float(np.mean(frame_speech[f_start:f_end]))
        c_rms = np.sqrt(np.mean(c ** 2) + 1e-9)
        rel_energy = float(np.clip((c_rms - 0.20 * primary_rms) / (0.40 * primary_rms + 1e-6), 0.02, 1.0))
        weights.append(speech_val * rel_energy)

    return np.stack(chunks).astype(np.float32), np.array(weights, dtype=np.float32)

def aggregate_salience_probabilities(chunk_probs: np.ndarray, chunk_weights: Optional[np.ndarray] = None, temperature: float = 0.5) -> np.ndarray:
    if len(chunk_probs) == 1:
        return chunk_probs[0]

    salience = np.max(chunk_probs, axis=1) - np.min(chunk_probs, axis=1)
    if chunk_weights is not None and len(chunk_weights) == len(chunk_probs):
        max_w = float(np.max(chunk_weights))
        norm_weights = (chunk_weights / max_w) if max_w > 1e-6 else chunk_weights
        salience = salience * np.clip(norm_weights, 0.05, 1.0)

    exp_weights = np.exp(salience / max(temperature, 1e-3))
    weights = exp_weights / np.sum(exp_weights)

    final_probs = np.sum(weights[:, np.newaxis] * chunk_probs, axis=0)
    return final_probs.astype(np.float32)

def preprocess_audio_pipeline(audio_bytes: bytes) -> Dict[str, Any]:
    raw_signal, sr = decode_audio_bytes(audio_bytes)

    duration = float(len(raw_signal) / sr)
    rms_energy = compute_rms(raw_signal)

    if duration < 0.5:
        return {
            "is_valid": False,
            "error": "File âm thanh quá ngắn (< 0.5s). Vui lòng cung cấp đoạn thu âm dài hơn.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    if rms_energy < 0.002:
        return {
            "is_valid": False,
            "error": "Không phát hiện tiếng người rõ ràng (Nhiễu nền hoặc im lặng, RMS < 0.002).",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    cleaned_signal, frame_speech, perc_ratio, snr_db, fund_ratio, formant_ratio = apply_adaptive_spectral_cleanup(raw_signal, sr=sr)
    voiced_ratio = compute_voicing_ratio(cleaned_signal, sr=sr)

    if formant_ratio < 0.20:
        return {
            "is_valid": False,
            "error": "Không phát hiện dải tần số giọng nói con người (Tín hiệu là tạp âm tần số thấp hoặc ù điện). Vui lòng cung cấp bản ghi có tiếng người nói.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    if fund_ratio < 0.05:
        return {
            "is_valid": False,
            "error": "Không phát hiện âm bội thanh quản giọng nói con người (Tín hiệu là âm nhạc điện tử, tiếng rít hoặc tạp âm môi trường). Vui lòng cung cấp bản ghi có tiếng người nói.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    if perc_ratio >= 0.40 and fund_ratio < 0.08:
        return {
            "is_valid": False,
            "error": "Không phát hiện tiếng nói con người rõ ràng trong tệp âm thanh (Phát hiện âm nhạc, tiết tấu trống hoặc beat không chứa giọng nói). Vui lòng cung cấp bản ghi có tiếng người nói.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    if voiced_ratio < 0.12 and fund_ratio < 0.10:
        return {
            "is_valid": False,
            "error": "Không phát hiện dao động thanh đới của giọng nói con người (Tạp âm ngẫu nhiên hoặc nhạc cụ nền). Vui lòng cung cấp bản ghi có tiếng người nói.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    if voiced_ratio < 0.08 and formant_ratio < 0.50:
        return {
            "is_valid": False,
            "error": "Không phát hiện tiếng nói con người rõ ràng (Nhiễu nền môi trường không có tiếng người). Vui lòng cung cấp bản ghi có tiếng người nói.",
            "duration": duration,
            "rms": rms_energy,
            "chunks": None,
            "chunk_weights": None,
            "num_chunks": 0
        }

    trimmed_signal = trim_silence_vad(cleaned_signal, frame_speech)
    signal_to_chunk = trimmed_signal if len(trimmed_signal) >= int(0.5 * TARGET_SR) else cleaned_signal

    chunks, chunk_weights = adaptive_overlapping_chunking(signal_to_chunk, frame_speech, sr=sr)

    return {
        "is_valid": True,
        "error": None,
        "chunks": chunks,
        "chunk_weights": chunk_weights,
        "duration": duration,
        "rms": rms_energy,
        "num_chunks": chunks.shape[0],
        "perc_ratio": round(perc_ratio, 3),
        "snr_db": round(snr_db, 2)
    }
