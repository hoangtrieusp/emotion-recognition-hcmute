import os
import sys
import io
import gc
import time
import psutil
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient
from servers.app.main import app

def generate_10s_audio_bytes() -> bytes:
    """Sinh file WAV 10 giây (160,000 mẫu ở 16kHz) có dạng sóng điều hòa giọng nói."""
    sr = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Giả lập tín hiệu giọng nói phong phú
    f0 = 200.0
    mod = 1.0 + 0.4 * np.sin(2 * np.pi * 3 * t)
    signal = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * (2 * f0) * t) +
        0.2 * np.sin(2 * np.pi * (3 * f0) * t) +
        0.05 * np.random.normal(0, 0.05, len(t))
    ) * mod
    signal = signal.astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    data = buf.getvalue()
    buf.close()
    return data

def get_process_memory_mb() -> float:
    """Lấy dung lượng RAM RSS (Resident Set Size) hiện tại của tiến trình bằng psutil."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_memory_leak_test():
    print("\n" + "="*70)
    print("  BẮT ĐẦU KIỂM THỬ PROFILING BỘ NHỚ RAM (50 REQUESTS - 10s AUDIO)")
    print("="*70)

    audio_bytes = generate_10s_audio_bytes()
    print(f"[*] Payload audio: 10.0s | Kích thước: {len(audio_bytes) / 1024:.2f} KB | Mẫu: 160,000 float32")

    with TestClient(app) as client:
        # 1. Warm-up 5 requests để ONNX Runtime cấp phát memory pool tĩnh ban đầu
        print("[*] Chạy 5 requests Warm-up để ổn định bộ nhớ ONNX Runtime...")
        for _ in range(5):
            client.post(
                "/api/v1/predict/emotion",
                files={"file": ("warmup.wav", audio_bytes, "audio/wav")}
            )

        gc.collect()
        time.sleep(0.5)

        baseline_ram = get_process_memory_mb()
        print(f"\n[BASELINE] RAM sau Warm-up: {baseline_ram:.2f} MB")
        print("-" * 70)
        print(f"{'Request':<10} | {'RAM (MB)':<12} | {'Delta vs Base':<15} | {'E2E Latency (ms)':<18}")
        print("-" * 70)

        history = []
        num_requests = 50

        for i in range(1, num_requests + 1):
            t0 = time.perf_counter()
            res = client.post(
                "/api/v1/predict/emotion",
                files={"file": (f"test_10s_{i}.wav", audio_bytes, "audio/wav")}
            )
            lat = (time.perf_counter() - t0) * 1000.0

            assert res.status_code == 200, f"Request {i} failed: {res.text}"
            
            # Ghi nhận bộ nhớ mỗi 5 requests và các request đầu/cuối
            if i % 5 == 0 or i == 1:
                current_ram = get_process_memory_mb()
                delta = current_ram - baseline_ram
                history.append((i, current_ram, delta, lat))
                print(f"#{i:<9} | {current_ram:<12.2f} | {delta:+<15.2f} | {lat:<18.1f}")

        # Thu gom rác để đo bộ nhớ thực sự bị giữ lại
        gc.collect()
        time.sleep(0.5)
        final_ram = get_process_memory_mb()
        total_delta = final_ram - baseline_ram

        print("-" * 70)
        print(f"[FINAL] RAM sau 50 requests & GC: {final_ram:.2f} MB (Tăng: {total_delta:+.2f} MB)")
        
        # Vẽ biểu đồ ASCII tiến trình RAM
        print("\n[*] BIỂU ĐỒ TIẾN TRÌNH RAM QUA CÁC MỐC REQUESTS:")
        min_r = min(h[1] for h in history)
        max_r = max(h[1] for h in history)
        span = max(1.0, max_r - min_r)

        for req_idx, ram, delta, _ in history:
            bar_len = int(((ram - min_r) / span) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"Req {req_idx:02d} [{bar}] {ram:.1f} MB ({delta:+.2f} MB)")

        print("\n" + "="*70)
        print("  ĐÁNH GIÁ CHỈ SỐ RÒ RỈ BỘ NHỚ (MEMORY LEAK ASSESSMENT)")
        print("="*70)
        print(f"  • RAM ban đầu: {baseline_ram:.2f} MB")
        print(f"  • RAM kết thúc: {final_ram:.2f} MB")
        print(f"  • Chênh lệch (Delta): {total_delta:+.2f} MB")
        print(f"  • Tỷ lệ rò rỉ trung bình mỗi request: {total_delta / num_requests:+.4f} MB/req")

        # Ngưỡng rò rỉ: Sau 50 requests 10s audio, nếu RAM tăng vượt quá 50MB (tức >1MB/request liên tục không nhả)
        # thì coi là rò rỉ bộ nhớ. Nếu chỉ dao động nhỏ (< 15MB) là do OS heap fragmentation bình thường.
        if total_delta > 50.0:
            print("  [CẢNH BÁO] Phát hiện dấu hiệu Rò Rỉ Bộ Nhớ (Memory Leak > 50MB)!")
            raise AssertionError(f"Memory leak detected: Delta RAM = {total_delta:.2f} MB > 50 MB")
        else:
            print("  [KẾT LUẬN] KHÔNG CÓ RÒ RỈ BỘ NHỚ (STABLE MEMORY FOOTPRINT)!")
            print("  --> Bộ nhớ giữ ở mức ổn định, GC hoạt động hiệu quả, bộ đệm được giải phóng đúng chuẩn.")
        print("="*70 + "\n")

if __name__ == "__main__":
    run_memory_leak_test()
