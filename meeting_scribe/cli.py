"""CLI entrypoint: orchestrates the transcription pipeline."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from . import transcribe as transcribe_mod
from . import diarize as diarize_mod
from .merge import merge, LabeledSegment
from . import translate as translate_mod


console = Console(stderr=True)


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), default=None, help="Output file (default: stdout)")
@click.option("--num-speakers", type=int, default=None, help="Hint speaker count to pyannote")
@click.option("--skip-translation", is_flag=True, help="Output raw multilingual transcript")
@click.option("--no-diarization", is_flag=True, help="Skip speaker diarization")
def main(
    audio_file: str,
    output: str | None,
    num_speakers: int | None,
    skip_translation: bool,
    no_diarization: bool,
) -> None:
    """Transcribe a meeting recording with speaker labels and translation."""
    audio_path = str(Path(audio_file).resolve())

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        if no_diarization:
            whisper_task = progress.add_task("Transcribing (Whisper GPU)", total=None)
            segments = transcribe_mod.transcribe(audio_path)
            progress.update(whisper_task, total=1, completed=1, description="[green]Transcription done")

            labeled = [
                LabeledSegment(s.start, s.end, "SPEAKER", s.text)
                for s in segments
            ]
        else:
            whisper_task = progress.add_task("Transcribing (Whisper GPU)", total=None)
            diarize_task = progress.add_task("Diarizing speakers (CPU)", total=None)

            # pyannote progress hook: update the rich progress bar
            diarize_steps_seen: dict[str, int] = {}

            def pyannote_hook(step_name, _artefact=None, completed=None, total=None, **_kw):
                if completed is not None and total is not None:
                    progress.update(diarize_task, total=total, completed=completed,
                                    description=f"Diarizing: {step_name}")
                elif step_name not in diarize_steps_seen:
                    diarize_steps_seen[step_name] = 1
                    progress.update(diarize_task, description=f"Diarizing: {step_name}")

            def run_whisper():
                result = transcribe_mod.transcribe(audio_path)
                progress.update(whisper_task, total=1, completed=1,
                                description=f"[green]Transcribed: {len(result)} segments")
                return result

            def run_diarize():
                result = diarize_mod.diarize(audio_path, num_speakers=num_speakers, hook=pyannote_hook)
                n_speakers = len(set(t.speaker for t in result))
                progress.update(diarize_task, total=1, completed=1,
                                description=f"[green]Diarized: {n_speakers} speakers")
                return result

            with ThreadPoolExecutor(max_workers=2) as pool:
                seg_future: Future = pool.submit(run_whisper)
                dia_future: Future = pool.submit(run_diarize)
                segments = seg_future.result()
                turns = dia_future.result()

            merge_task = progress.add_task("Merging segments", total=1)
            labeled = merge(segments, turns)
            progress.update(merge_task, completed=1,
                            description=f"[green]Merged: {len(labeled)} segments")

        # Translation
        if skip_translation:
            result = translate_mod.format_raw(labeled)
        else:
            translate_task = progress.add_task("Translating via Claude", total=None)

            def on_chunk_done(completed: int, total: int) -> None:
                if completed < total:
                    progress.update(translate_task, total=total, completed=completed,
                                    description=f"Translating ({completed}/{total - 1} chunks)")
                else:
                    progress.update(translate_task, total=total, completed=completed,
                                    description="Reviewing translation")

            result = translate_mod.translate(labeled, on_chunk_done=on_chunk_done)
            progress.update(translate_task, total=1, completed=1,
                            description="[green]Translation done (2-pass)")

    # Output
    if output:
        Path(output).write_text(result + "\n")
        console.print(f"[green]Written to {output}")
    else:
        click.echo(result)
