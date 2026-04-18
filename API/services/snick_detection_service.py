from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import signal


@dataclass
class AudioAnalysisConfig:
    low_hz: int = 1200
    high_hz: int = 6500
    peak_prominence: float = 2.5
    min_peak_distance_ms: int = 30
    align_window_ms: int = 80


class SnickDetectionService:
    """Lightweight audio transient detector for bat-ball edge hints."""

    @staticmethod
    def _read_wav_mono(audio_wav_path: str) -> tuple[np.ndarray, int]:
        p = Path(audio_wav_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_wav_path}")

        with wave.open(str(p), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sample_width == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)

        return audio, int(sample_rate)

    @staticmethod
    def _bandpass_filter(audio: np.ndarray, sr: int, low_hz: int, high_hz: int) -> np.ndarray:
        nyq = sr / 2.0
        low = max(20.0, min(low_hz, nyq - 50.0)) / nyq
        high = max(low + 1e-3, min(high_hz, nyq - 1.0)) / nyq
        b, a = signal.butter(4, [low, high], btype="bandpass")
        return signal.filtfilt(b, a, audio)

    @staticmethod
    def _transient_score_series(audio: np.ndarray, sr: int) -> np.ndarray:
        # Envelope plus differentiated energy highlights short, sharp snick-like events.
        env = np.abs(audio)
        win = max(1, int(0.003 * sr))
        kernel = np.ones(win, dtype=np.float32) / float(win)
        smooth = np.convolve(env, kernel, mode="same")
        deriv = np.maximum(0.0, np.diff(smooth, prepend=smooth[0]))
        score = 0.7 * smooth + 0.3 * deriv
        return score.astype(np.float32)

    @staticmethod
    def _robust_normalize(x: np.ndarray) -> np.ndarray:
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        scale = max(1e-6, mad * 1.4826)
        return (x - med) / scale

    @staticmethod
    def analyze(
        audio_wav_path: str,
        *,
        fps: int,
        preferred_frame_idx: Optional[int] = None,
        config: Optional[AudioAnalysisConfig] = None,
    ) -> Dict[str, Any]:
        cfg = config or AudioAnalysisConfig()
        audio, sr = SnickDetectionService._read_wav_mono(audio_wav_path)
        if audio.size == 0:
            return {
                "status": "unavailable",
                "reason": "empty_audio",
                "snick_detected": False,
                "audio_confidence": 0.0,
            }

        filtered = SnickDetectionService._bandpass_filter(audio, sr, cfg.low_hz, cfg.high_hz)
        score = SnickDetectionService._transient_score_series(filtered, sr)
        z = SnickDetectionService._robust_normalize(score)

        min_peak_distance = max(1, int((cfg.min_peak_distance_ms / 1000.0) * sr))
        peak_indices, props = signal.find_peaks(z, prominence=cfg.peak_prominence, distance=min_peak_distance)

        peaks: List[Dict[str, Any]] = []
        prominences = props.get("prominences", np.array([], dtype=np.float32))
        for i, idx in enumerate(peak_indices):
            ts_ms = float(idx) * 1000.0 / float(sr)
            prom = float(prominences[i]) if i < len(prominences) else 0.0
            conf = 1.0 / (1.0 + math.exp(-(prom - 2.5)))
            peaks.append(
                {
                    "sample_idx": int(idx),
                    "timestamp_ms": ts_ms,
                    "prominence": prom,
                    "confidence": float(max(0.0, min(1.0, conf))),
                }
            )

        target_ms = None
        if preferred_frame_idx is not None and fps > 0:
            target_ms = (float(preferred_frame_idx) / float(fps)) * 1000.0

        best_peak = None
        aligned = False
        if peaks and target_ms is not None:
            best_peak = min(peaks, key=lambda p: abs(p["timestamp_ms"] - target_ms))
            delta_ms = abs(best_peak["timestamp_ms"] - target_ms)
            aligned = delta_ms <= float(cfg.align_window_ms)
            if not aligned:
                best_peak = None

        if best_peak is None and peaks:
            best_peak = max(peaks, key=lambda p: p["confidence"])

        audio_conf = float(best_peak["confidence"]) if best_peak else 0.0
        detected = audio_conf >= 0.6

        return {
            "status": "ok",
            "reason": None,
            "snick_detected": bool(detected),
            "audio_confidence": audio_conf,
            "best_event_timestamp_ms": float(best_peak["timestamp_ms"]) if best_peak else None,
            "target_timestamp_ms": target_ms,
            "aligned_with_target": bool(aligned) if target_ms is not None else None,
            "candidate_event_count": len(peaks),
            "top_events": sorted(peaks, key=lambda p: p["confidence"], reverse=True)[:5],
        }
