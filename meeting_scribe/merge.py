"""Merge Whisper segments with pyannote speaker turns by temporal overlap."""

from __future__ import annotations

from dataclasses import dataclass

from .transcribe import Segment
from .diarize import SpeakerTurn


@dataclass
class LabeledSegment:
    start: float
    end: float
    speaker: str
    text: str


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return duration of overlap between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge(
    segments: list[Segment],
    turns: list[SpeakerTurn],
) -> list[LabeledSegment]:
    """Assign a speaker to each transcript segment based on greatest overlap."""
    labeled = []
    for seg in segments:
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in turns:
            ov = _overlap(seg.start, seg.end, turn.start, turn.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = turn.speaker
        labeled.append(LabeledSegment(
            start=seg.start,
            end=seg.end,
            speaker=best_speaker,
            text=seg.text,
        ))
    return labeled
