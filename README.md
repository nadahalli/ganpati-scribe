# ganpati-scribe

Local multilingual meeting transcription with speaker diarization and English translation.

Built for transcribing meetings with mixed languages (Swiss German, German, English, French). Heavy compute runs locally on Apple Silicon, Claude API handles translation only.

## Pipeline

```
audio file
   |
   +---> mlx-whisper (Metal GPU)      --> timestamped text segments
   |     whisper-large-v3-turbo
   |
   +---> pyannote.audio (CPU)          --> speaker timeline
   |     speaker-diarization-3.1
   |
   +---> Merge by timecodes            --> speaker-labeled segments
   |
   +---> Claude API (two-pass)         --> final English transcript
         pass 1: translate chunks
         pass 2: review consistency, merge speaker turns
```

Steps 1 and 2 run in parallel.

## Setup

**Prerequisites:**
- Python 3.12
- ffmpeg (`brew install ffmpeg`)
- [HuggingFace token](https://hf.co/settings/tokens) with access to these gated models:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Anthropic API key](https://console.anthropic.com) (for translation)

**Install:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Environment variables:**
```bash
export HF_TOKEN=hf_...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
meeting-scribe meeting.mp4                     # full pipeline, output to stdout
meeting-scribe meeting.mp4 -o transcript.txt   # write to file
meeting-scribe meeting.mp4 --num-speakers 4    # hint speaker count
meeting-scribe meeting.mp4 --skip-translation  # raw multilingual transcript (100% local)
meeting-scribe meeting.mp4 --no-diarization    # skip speaker labels
```

## Performance (M4 Max, 48GB)

For a 1-hour recording, expect ~5-8 minutes total (whisper and diarization run in parallel).

`--skip-translation` runs entirely offline with no API calls.
