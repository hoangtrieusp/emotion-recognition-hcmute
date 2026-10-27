import os
import sys
import io
import subprocess
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath("."))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi.testclient import TestClient
from servers.app.main import app

def generate_webm_bytes(duration: float = 3.0) -> bytes:
    """Sử dụng ffmpeg sinh ra luồng byte chuẩn audio/webm (Opus) giả lập MediaRecorder từ Browser."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=300:duration={duration}",
        "-c:a", "libopus",
        "-f", "webm",
        "pipe:1"
    ]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return process.stdout

def test_resilience():
    print("\n" + "="*70)
    print("  BẮT ĐẦU CHẠY BỘ KIỂM ĐỊNH ĐỘ BỀN HỆ THỐNG (RESILIENCE & EDGE-CASES)")
    print("="*70)

    with TestClient(app) as client:
        # =========================================================================
        # KỊCH BẢN 1: FILE ĐỊNH DẠNG LẠ & TRỐNG (CORRUPTED & EMPTY PAYLOAD)
        # =========================================================================
        print("\n[KỊCH BẢN 1] Kiểm thử File định dạng lạ & Payload rỗng:")

        # 1A: File đuôi .wav nhưng nội dung là plain text giả mạo
        corrupted_bytes = b"THIS_IS_NOT_A_VALID_WAV_FILE_JUST_PLAIN_TEXT_FOR_TESTING_PURPOSE"
        res_1a = client.post(
            "/api/v1/predict/emotion",
            files={"file": ("fake_audio.wav", corrupted_bytes, "audio/wav")}
        )
        print(f"  1A. File .wav chứa text thuần: HTTP {res_1a.status_code} | Body: {res_1a.text}")
        assert res_1a.status_code == 400, f"Expected 400 Bad Request, got {res_1a.status_code}"
        assert "Không thể giải mã" in res_1a.text or "sai định dạng" in res_1a.text
        print("      --> PASS: Gateway chặn đứng file giả mạo với lỗi 400 Bad Request (Zero crash).")

        # 1B: File rỗng 0 bytes
        empty_bytes = b""
        res_1b = client.post(
            "/api/v1/predict/emotion",
            files={"file": ("empty.wav", empty_bytes, "audio/wav")}
        )
        print(f"  1B. File rỗng 0 bytes: HTTP {res_1b.status_code} | Body: {res_1b.text}")
        assert res_1b.status_code == 400, f"Expected 400 Bad Request, got {res_1b.status_code}"
        assert "File âm thanh rỗng" in res_1b.text
        print("      --> PASS: Gateway phát hiện payload 0 byte và từ chối 400 Bad Request.")

        # =========================================================================
        # KỊCH BẢN 2: BẪY RMS ENERGY (SILENT ATTACK)
        # =========================================================================
        print("\n[KỊCH BẢN 2] Kiểm thử Bẫy Năng lượng RMS (Silent Attack):")
        
        # Tạo file WAV dài 5 giây chứa sự im lặng hoàn toàn (RMS = 0.0)
        sr = 16000
        duration = 5.0
        silent_signal = np.zeros(int(sr * duration), dtype=np.float32)
        
        silent_buf = io.BytesIO()
        sf.write(silent_buf, silent_signal, sr, format="WAV")
        silent_buf.seek(0)

        res_2 = client.post(
            "/api/v1/predict/emotion",
            files={"file": ("silent_attack.wav", silent_buf.read(), "audio/wav")}
        )
        print(f"  2. File 5s im lặng tuyệt đối (RMS=0): HTTP {res_2.status_code}")
        assert res_2.status_code == 200, f"Expected 200 with warning JSON, got {res_2.status_code}"
        res_2_json = res_2.json()
        print(f"     Body: {res_2_json}")
        assert res_2_json["status"] == "warning", f"Expected warning status, got {res_2_json['status']}"
        assert res_2_json["dominant_emotion"] is None, "Dominant emotion must be None on silence rejection"
        assert "Không phát hiện tiếng người rõ ràng" in res_2_json["error"]
        assert res_2_json["telemetry"]["rms"] < 0.002
        print(f"      --> PASS: Hệ thống chối bỏ suy luận (RMS={res_2_json['telemetry']['rms']} < 0.002) đúng như thiết kế.")

        # =========================================================================
        # KỊCH BẢN 3: BẪY WEBM/OPUS (BROWSER MEDIARECORDER TEST)
        # =========================================================================
        print("\n[KỊCH BẢN 3] Kiểm thử Định dạng WebM/Opus từ Trình duyệt Browser:")
        
        webm_bytes = generate_webm_bytes(duration=3.0)
        print(f"  3. Luồng byte WebM/Opus sinh ra bởi FFmpeg: Dung lượng {len(webm_bytes)} bytes")
        
        res_3 = client.post(
            "/api/v1/predict/emotion",
            files={"file": ("recording_mic.webm", webm_bytes, "audio/webm")}
        )
        print(f"  3. Kết quả gọi API với file WebM/Opus: HTTP {res_3.status_code}")
        assert res_3.status_code == 200, f"Expected 200 OK, got {res_3.status_code}: {res_3.text}"
        res_3_json = res_3.json()
        print(f"     Body: {res_3_json}")
        assert res_3_json["status"] == "success", f"Expected success status, got {res_3_json['status']}"
        assert res_3_json["dominant_emotion"] in ["angry", "happy", "neutral", "sad"]
        assert res_3_json["telemetry"]["duration"] >= 2.8
        assert res_3_json["telemetry"]["inference_latency_ms"] > 0
        print(f"      --> PASS: Lõi DSP giải mã WebM/Opus thành công, suy luận ONNX E2E = {res_3_json['telemetry']['total_latency_ms']} ms.")

        print("\n" + "="*70)
        print("  [KẾT LUẬN] TẤT CẢ 3 KỊCH BẢN STRESS-TEST ĐỀU VƯỢT QUA VỚI EXIT CODE 0!")
        print("="*70 + "\n")

if __name__ == "__main__":
    test_resilience()
