#!/usr/bin/env python3
"""
Hurrian Hymn No. 6 (~1400 BCE) – oldest substantially complete notated song.
Rendered with three fluid-flow timbres at a comfortable listening range,
plus every pairwise mix and the full three-way mix.

Note: The original tablet notation is highly ambiguous. This uses a
simplified, widely performed reconstruction (Dumbrill-inspired, monophonic
melody in a diatonic mode) suitable for pure synthesis.

Output (./flow_sounds/):
  01_hurrian_laminar.wav
  02_hurrian_oscillating.wav
  03_hurrian_turbulent.wav
  04_hurrian_laminar_oscillating_mix.wav
  05_hurrian_laminar_turbulent_mix.wav
  06_hurrian_oscillating_turbulent_mix.wav
  07_hurrian_all_three_mix.wav
"""

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
AMPLITUDE = 0.38
OUTPUT_DIR = Path(__file__).parent / "flow_sounds"
OUTPUT_DIR.mkdir(exist_ok=True)

# Approximate concert pitch for the melody (comfortable hearing range)
# Scale centred near 300–500 Hz (well inside human peak sensitivity)
NOTE_FREQS = {
    "C4": 261.63,
    "D4": 293.66,
    "E4": 329.63,
    "F4": 349.23,
    "G4": 392.00,
    "A4": 440.00,
    "B4": 493.88,
    "C5": 523.25,
    "D5": 587.33,
    "E5": 659.25,
    "F5": 698.46,
    "G5": 783.99,
}

# Simplified Dumbrill-inspired melodic sequence (note, duration in beats)
# Tempo ≈ 90 BPM → beat = 0.666 s (echo of the original 666 Hz theme)
BEAT = 0.55  # seconds per beat
MELODY = [
    # Opening phrase
    ("E4", 1.5), ("G4", 1.0), ("A4", 1.0), ("G4", 0.5),
    ("E4", 1.5), ("D4", 1.0), ("E4", 2.0),
    # Second phrase
    ("G4", 1.0), ("A4", 1.0), ("B4", 1.5), ("A4", 0.5),
    ("G4", 1.0), ("E4", 1.5), ("D4", 1.5),
    # Third phrase
    ("E4", 1.0), ("G4", 1.0), ("A4", 1.0), ("C5", 1.5),
    ("B4", 1.0), ("A4", 1.0), ("G4", 2.0),
    # Closing
    ("A4", 1.0), ("G4", 1.0), ("E4", 1.5), ("D4", 1.0),
    ("E4", 3.0),
]


def normalize(signal: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * AMPLITUDE
    return signal.astype(np.float32)


def save_wav(filename: str, signal: np.ndarray) -> None:
    path = OUTPUT_DIR / filename
    pcm = np.int16(np.clip(signal, -1.0, 1.0) * 32767)
    wavfile.write(path, SAMPLE_RATE, pcm)
    print(f"  Saved: {path.name}  ({len(signal)/SAMPLE_RATE:.1f} s)")


# ---------------------------------------------------------------------------
# Timbre generators (same spirit as previous flow scripts)
# ---------------------------------------------------------------------------
def tone_laminar(freq: float, duration: float) -> np.ndarray:
    """Pure, smooth sine – laminar."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Soft attack / release envelope
    env = np.ones(n)
    attack = int(0.02 * SAMPLE_RATE)
    release = int(0.06 * SAMPLE_RATE)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return np.sin(2 * np.pi * freq * t) * env


def tone_oscillating(freq: float, duration: float) -> np.ndarray:
    """Amplitude-modulated – oscillating flow."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    carrier = np.sin(2 * np.pi * freq * t)
    mod = 1.0 - 0.55 * (0.5 + 0.5 * np.sin(2 * np.pi * 4.5 * t))
    env = np.ones(n)
    attack = int(0.02 * SAMPLE_RATE)
    release = int(0.06 * SAMPLE_RATE)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return carrier * mod * env


def tone_turbulent(freq: float, duration: float) -> np.ndarray:
    """Noisy + jitter – turbulent flow."""
    n = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    rng = np.random.default_rng(int(freq * 100) % 2**31)

    jitter = 0.025 * freq * rng.standard_normal(n)
    jitter = gaussian_filter1d(jitter, sigma=SAMPLE_RATE * 0.006)
    phase = 2 * np.pi * np.cumsum(freq + jitter) / SAMPLE_RATE
    carrier = np.sin(phase)

    noise = rng.standard_normal(n)
    noise = np.convolve(noise, np.ones(24) / 24, mode="same") * 0.35

    amp_mod = 0.55 + 0.45 * np.sin(2 * np.pi * 2.8 * t)
    amp_mod += 0.25 * gaussian_filter1d(rng.standard_normal(n), sigma=SAMPLE_RATE * 0.04)
    amp_mod = np.clip(amp_mod, 0.2, 1.0)

    env = np.ones(n)
    attack = int(0.02 * SAMPLE_RATE)
    release = int(0.06 * SAMPLE_RATE)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)

    return (carrier + noise) * amp_mod * env


def render_melody(tone_fn) -> np.ndarray:
    """Render the full melody with the given timbre function."""
    parts = []
    for note, beats in MELODY:
        freq = NOTE_FREQS[note]
        dur = beats * BEAT
        parts.append(tone_fn(freq, dur))
        # tiny gap between notes for clarity
        parts.append(np.zeros(int(0.015 * SAMPLE_RATE), dtype=np.float32))
    return normalize(np.concatenate(parts))


def mix(*signals: np.ndarray) -> np.ndarray:
    # Pad to same length
    max_len = max(len(s) for s in signals)
    padded = [np.pad(s, (0, max_len - len(s))) for s in signals]
    return normalize(np.sum(padded, axis=0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Rendering Hurrian Hymn No. 6 with flow timbres …\n")

    laminar     = render_melody(tone_laminar)
    oscillating = render_melody(tone_oscillating)
    turbulent   = render_melody(tone_turbulent)

    print("Single timbres:")
    save_wav("01_hurrian_laminar.wav", laminar)
    save_wav("02_hurrian_oscillating.wav", oscillating)
    save_wav("03_hurrian_turbulent.wav", turbulent)

    print("\nPairwise mixes:")
    save_wav("04_hurrian_laminar_oscillating_mix.wav",
             mix(laminar, oscillating))
    save_wav("05_hurrian_laminar_turbulent_mix.wav",
             mix(laminar, turbulent))
    save_wav("06_hurrian_oscillating_turbulent_mix.wav",
             mix(oscillating, turbulent))

    print("\nAll three mixed:")
    save_wav("07_hurrian_all_three_mix.wav",
             mix(laminar, oscillating, turbulent))

    print(f"\nFiles written to: {OUTPUT_DIR.resolve()}")
    print(
        "\nMelody is a simplified Dumbrill-inspired reconstruction of\n"
        "Hurrian Hymn No. 6 (oldest substantially complete notated song).\n"
        "Exact original pitches/rhythm remain scholarly contested."
    )
