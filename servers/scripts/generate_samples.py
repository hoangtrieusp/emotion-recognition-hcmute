import os
import numpy as np
import soundfile as sf

def generate_samples(output_dir: str = "clients/public/samples"):
    os.makedirs(output_dir, exist_ok=True)
    sr = 16000
    
    # 1. Angry Sample (High energy, rapid aggressive modulation, bright harmonics)
    dur = 2.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0 = 320.0
    mod = 1.0 + 0.5 * np.sin(2 * np.pi * 12 * t)
    audio_angry = (
        0.5 * np.sin(2 * np.pi * f0 * t) +
        0.3 * np.sin(2 * np.pi * (2 * f0) * t) +
        0.2 * np.sin(2 * np.pi * (3 * f0) * t) +
        0.05 * np.random.normal(0, 0.1, len(t))
    ) * mod
    audio_angry = np.clip(audio_angry, -0.9, 0.9).astype(np.float32)
    sf.write(os.path.join(output_dir, "angry_sample.wav"), audio_angry, sr)

    # 2. Happy Sample (Upward inflection, warm melodic contour)
    dur = 2.8
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f_contour = 240 + 80 * np.sin(2 * np.pi * 1.5 * t)
    phase = 2 * np.pi * np.cumsum(f_contour) / sr
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t)) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
    audio_happy = (np.sin(phase) + 0.4 * np.sin(2 * phase)) * env
    audio_happy = np.clip(audio_happy, -0.85, 0.85).astype(np.float32)
    sf.write(os.path.join(output_dir, "happy_sample.wav"), audio_happy, sr)

    # 3. Neutral Sample (Monotone, flat prosody, steady volume)
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0 = 160.0
    env = 0.4 * np.ones_like(t)
    audio_neutral = (np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)) * env
    audio_neutral = audio_neutral.astype(np.float32)
    sf.write(os.path.join(output_dir, "neutral_sample.wav"), audio_neutral, sr)

    # 4. Sad Sample (Low pitch, soft downward glide, low energy)
    dur = 3.2
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f_contour = 130 - 25 * (t / dur)
    phase = 2 * np.pi * np.cumsum(f_contour) / sr
    env = 0.25 * np.exp(-0.8 * t) * (0.6 + 0.4 * np.cos(2 * np.pi * 1.2 * t))
    audio_sad = (np.sin(phase) + 0.15 * np.sin(2 * phase)) * env
    audio_sad = audio_sad.astype(np.float32)
    sf.write(os.path.join(output_dir, "sad_sample.wav"), audio_sad, sr)

    print(f"[SUCCESS] Created 4 sample audio files in '{output_dir}'.")

if __name__ == "__main__":
    generate_samples()
