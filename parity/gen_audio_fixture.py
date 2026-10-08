#!/usr/bin/env python3
"""Generate the audio parity fixtures for the mtmd-audio port.

Writes (deterministic, stdlib only):
    parity/mtmd-fixture-audio.wav — 16-bit PCM mono 16 kHz, ~2.56 s of a
        deterministic multi-tone + ramp signal (the WAV the port's reader and
        the reference miniaudio path both decode)
    parity/mtmd-fixture-audio.f32 — the exact f32 samples [-1, 1) the WAV
        decodes to (16-bit >> 15), i.e. the ground-truth PCM the reference
        probe (parity/ref_mtmd_audio_dump.cpp) preprocesses
    parity/mtmd-fixture-audio-24k.wav — the same signal at 24 kHz (audio
        round 4: the mimo / qwen3tts_spkenc / pockettts_spkenc preprocessors
        run at 24 kHz; a rate-matched WAV is the only input the port's
        no-resample reader accepts, and a rate-matched file is what makes the
        reference miniaudio path a passthrough)
"""

import math
import os
import struct


def signal(i: int, sr: int, n: int) -> float:
    x = i / sr
    v = (
        0.30 * math.sin(2 * math.pi * 220.0 * x)
        + 0.20 * math.sin(2 * math.pi * 587.0 * x + 0.7)
        + 0.10 * math.sin(2 * math.pi * 1330.0 * x + 1.9)
        + 0.05 * (i % 640) / 640.0
    )
    # an amplitude ramp so preemphasis/normalization see changing energy
    v *= 0.5 + 0.5 * (i / n)
    return v


def write_wav(path: str, sr: int, n: int) -> list:
    samples = [signal(i, sr, n) for i in range(n)]
    pcm = [max(-32768, min(32767, int(round(s * 32767.0)))) for s in samples]
    with open(path, "wb") as f:
        data = struct.pack("<" + "h" * len(pcm), *pcm)
        # RIFF: PCM mono 16-bit
        byte_rate = sr * 1 * 2
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data)))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, byte_rate, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", len(data)))
        f.write(data)
    return pcm


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    pcm16 = write_wav(os.path.join(here, "mtmd-fixture-audio.wav"), 16000, 40960)
    # the f32 the WAV decodes to: int16 / 32768 (miniaudio's i16 -> f32)
    with open(os.path.join(here, "mtmd-fixture-audio.f32"), "wb") as f:
        f.write(struct.pack("<" + "f" * len(pcm16), *(p / 32768.0 for p in pcm16)))
    print(f"mtmd-fixture-audio.wav: {len(pcm16)} samples @ 16000 Hz ({len(pcm16) / 16000:.2f} s)")

    pcm24 = write_wav(os.path.join(here, "mtmd-fixture-audio-24k.wav"), 24000, 61440)
    print(f"mtmd-fixture-audio-24k.wav: {len(pcm24)} samples @ 24000 Hz ({len(pcm24) / 24000:.2f} s)")
    print(f"mtmd-fixture-audio.f32: {len(pcm16)} f32 samples")


if __name__ == "__main__":
    main()
