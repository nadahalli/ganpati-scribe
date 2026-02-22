# ganpati-scribe

Local multilingual meeting transcription with speaker diarization and English translation.

Built for transcribing meetings with mixed languages (Swiss German, German, English, French). Heavy compute runs locally, Claude API handles translation only.

## Pipeline

```
audio file
   |
   +---> Whisper large-v3-turbo        --> timestamped text segments
   |     mlx-whisper (macOS Metal GPU)
   |     faster-whisper (Linux CPU)
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
- ffmpeg
- [HuggingFace token](https://hf.co/settings/tokens) with access to these gated models:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Anthropic API key](https://console.anthropic.com) (for translation)

**Install (macOS, Apple Silicon):**
```bash
brew install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[mac]"
```

**Install (Linux):**
```bash
sudo apt install ffmpeg    # or: dnf install ffmpeg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[linux]"
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

## Performance

| Platform | 1-hour recording |
|----------|-----------------|
| M4 Max (48GB) | ~5-8 min |
| AMD Ryzen 8-core (98GB) | ~10-20 min (CPU-bound whisper) |

Whisper and diarization run in parallel, so wall clock is max(whisper, pyannote) + merge + translate.

`--skip-translation` runs entirely offline with no API calls.
