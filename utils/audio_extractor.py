import shutil
import subprocess
from pathlib import Path
from typing import Optional


class AudioExtractionError(RuntimeError):
    """Raised when audio extraction from a video file fails."""


def resolve_ffmpeg_binary(explicit_binary: Optional[str] = None) -> Optional[str]:
    """Return a usable ffmpeg executable path, preferring explicit config then PATH."""
    if explicit_binary:
        p = Path(explicit_binary)
        if p.exists():
            return str(p)
        found = shutil.which(explicit_binary)
        if found:
            return found

    return shutil.which("ffmpeg")


def extract_audio_to_wav(
    video_path: str,
    *,
    output_wav_path: Optional[str] = None,
    sample_rate: int = 16000,
    ffmpeg_binary: Optional[str] = None,
) -> str:
    """
    Extract mono WAV audio from a video file using ffmpeg.

    Returns absolute path to the extracted WAV.
    """
    source = Path(video_path)
    if not source.exists():
        raise AudioExtractionError(f"Video file not found: {video_path}")

    ffmpeg_bin = resolve_ffmpeg_binary(ffmpeg_binary)
    if not ffmpeg_bin:
        raise AudioExtractionError(
            "FFmpeg binary not found. Install ffmpeg or set FFMPEG_BINARY in environment."
        )

    if output_wav_path:
        output = Path(output_wav_path)
    else:
        output = source.with_suffix(".wav")

    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-acodec",
        "pcm_s16le",
        str(output),
    ]

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if completed.returncode != 0 or not output.exists() or output.stat().st_size == 0:
        detail = (completed.stderr or completed.stdout or "unknown ffmpeg error").strip()
        raise AudioExtractionError(f"FFmpeg extraction failed: {detail}")

    return str(output.resolve())


def is_ffmpeg_available(explicit_binary: Optional[str] = None) -> bool:
    return resolve_ffmpeg_binary(explicit_binary) is not None
