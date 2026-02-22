"""Whisper transcription: mlx-whisper on macOS (Metal GPU), faster-whisper on Linux (CPU)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _transcribe_mlx(audio_path: str) -> list[Segment]:
    import mlx_whisper

    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
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


def _transcribe_faster(audio_path: str) -> list[Segment]:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        "large-v3-turbo",
        device="cpu",
        compute_type="int8",
    )
    result_segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        condition_on_previous_text=False,
        language=None,
        task="transcribe",
    )
    segments = []
    for seg in result_segments:
        segments.append(Segment(
            start=seg.start,
            end=seg.end,
            text=seg.text.strip(),
        ))
    return segments


def transcribe(audio_path: str) -> list[Segment]:
    """Transcribe audio, returning timestamped segments in original language.

    Automatically selects mlx-whisper (macOS) or faster-whisper (Linux).
    """
    try:
        import mlx_whisper  # noqa: F401
        return _transcribe_mlx(audio_path)
    except ImportError:
        pass

    try:
        import faster_whisper  # noqa: F401
        return _transcribe_faster(audio_path)
    except ImportError:
        raise RuntimeError(
            "No whisper backend found. Install mlx-whisper (macOS) or faster-whisper (Linux):\n"
            "  pip install meeting-scribe[mac]   # macOS with Apple Silicon\n"
            "  pip install meeting-scribe[linux]  # Linux"
        )
