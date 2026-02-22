"""Speaker diarization via pyannote.audio (CPU only, MPS broken)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
from pyannote.audio import Pipeline


MODEL = "pyannote/speaker-diarization-3.1"


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str


def diarize(
    audio_path: str,
    num_speakers: int | None = None,
    hook: Callable | None = None,
) -> list[SpeakerTurn]:
    """Run speaker diarization, returning speaker-labeled time intervals.

    hook: optional callback matching pyannote's signature:
        hook(step_name, step_artefact, completed=N, total=M, file=...)
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN env var required for pyannote model access")

    pipeline = Pipeline.from_pretrained(MODEL, token=hf_token)
    pipeline.to(torch.device("cpu"))

    params = {}
    if num_speakers is not None:
        params["num_speakers"] = num_speakers
    if hook is not None:
        params["hook"] = hook

    output = pipeline(audio_path, **params)

    # pyannote 4.x returns DiarizeOutput; extract the Annotation
    annotation = getattr(output, "speaker_diarization", output)

    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(
            start=turn.start,
            end=turn.end,
            speaker=speaker,
        ))
    return turns
