"""Whisper transcription via mlx-whisper (Metal GPU)."""

from __future__ import annotations

from dataclasses import dataclass

import mlx_whisper


MODEL = "mlx-community/whisper-large-v3-turbo"


@dataclass
class Segment:
    start: float
    end: float
    text: str


def transcribe(audio_path: str) -> list[Segment]:
    """Transcribe audio, returning timestamped segments in original language.

    Uses verbose=False to enable mlx-whisper's built-in tqdm progress bar.
    """
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=MODEL,
        verbose=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        language=None,
        task="transcribe",
    )
    segments = []
    for seg in result.get("segments", []):
        segments.append(Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"].strip(),
        ))
    return segments
