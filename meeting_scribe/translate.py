"""Translate non-English segments to English via Claude API."""

from __future__ import annotations

import os
from typing import Callable

import anthropic

from .merge import LabeledSegment


CHUNK_SIZE = 50

TRANSLATE_SYSTEM = """\
You are a meeting transcript translator. You receive segments of a multilingual \
meeting transcript with speaker labels and timestamps.

Your job:
- Translate all non-English text to English.
- Swiss German (Schwiizerdütsch) should be interpreted charitably and translated \
to natural English. It will often be phonetic or dialect spelling.
- Standard German and French should be translated accurately.
- English segments with heavy accents may have transcription errors; fix obvious \
ones but do not rephrase.
- Preserve speaker labels and timestamps exactly as given.
- Do NOT summarize. Output every segment, translated verbatim.
- Output format: one segment per line, same format as input."""

REVIEW_SYSTEM = """\
You are a transcript reviewer. You receive a full translated meeting transcript \
(pass 1) and must produce a cleaned-up final version (pass 2).

Your job:
- Fix inconsistent translations: if the same Swiss German word or phrase was \
translated differently in different places, pick the best translation and use it \
consistently throughout.
- Fix obvious translation errors or awkward phrasing from pass 1.
- Merge consecutive lines from the same speaker into a single block. Keep the \
timestamp range (use the earliest start and latest end).
- Clean up filler or repeated words that are artifacts of speech-to-text.
- Preserve all speaker labels and timestamp ranges.
- Do NOT summarize or omit any content. Every piece of information must be kept.
- Output format: one segment per line, same format as input."""


def _format_segments(segments: list[LabeledSegment]) -> str:
    lines = []
    for seg in segments:
        ts = f"[{seg.start:.1f}-{seg.end:.1f}]"
        lines.append(f"{ts} {seg.speaker}: {seg.text}")
    return "\n".join(lines)


def _parse_response(text: str) -> list[str]:
    """Return translated lines, filtering blanks."""
    return [line for line in text.strip().splitlines() if line.strip()]


def translate(
    segments: list[LabeledSegment],
    on_chunk_done: Callable[[int, int], None] | None = None,
) -> str:
    """Two-pass translation: translate in chunks, then review for consistency.

    on_chunk_done: optional callback(completed, total) called after each step.
    Pass 1 chunks count as steps 1..N, pass 2 review counts as step N+1.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY env var required for translation")

    client = anthropic.Anthropic(api_key=api_key)

    # Pass 1: chunked translation
    all_lines: list[str] = []
    num_chunks = (len(segments) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_steps = num_chunks + 1  # +1 for review pass

    for chunk_idx, i in enumerate(range(0, len(segments), CHUNK_SIZE)):
        chunk = segments[i : i + CHUNK_SIZE]
        formatted = _format_segments(chunk)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=TRANSLATE_SYSTEM,
            messages=[{"role": "user", "content": formatted}],
        )
        response_text = message.content[0].text
        all_lines.extend(_parse_response(response_text))

        if on_chunk_done:
            on_chunk_done(chunk_idx + 1, total_steps)

    pass1_transcript = "\n".join(all_lines)

    # Pass 2: review for consistency and merge same-speaker runs
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=REVIEW_SYSTEM,
        messages=[{"role": "user", "content": pass1_transcript}],
    )
    final = message.content[0].text.strip()

    if on_chunk_done:
        on_chunk_done(total_steps, total_steps)

    return final


def format_raw(segments: list[LabeledSegment]) -> str:
    """Format segments without translation."""
    return _format_segments(segments)
