## ytpipe

YouTube audio download and transcription pipeline.

- **Download** audio from YouTube (videos, playlists, channels) via `yt-dlp`
- **Transcribe** locally using `faster-whisper` (CPU or NVIDIA GPU)
- **CLI-first** with clean Python API
- Sensible outputs: audio + info JSON, a line-delimited manifest, and transcript JSON/TXT


## Features

- **One-liner pipeline**: `ytpipe pipeline -s <url> --out data`
- **Robust downloads** powered by `yt-dlp` with per-item info JSON and manifest
- **Fast transcription** via `faster-whisper`, automatic device/precision selection
- **Clear structure**: `data/raw/audio/*.{m4a,info.json}`, transcripts in `data/transcripts`


## Requirements

- Python >= 3.10
- `ffmpeg` available (in `PATH` or provided via `--ffmpeg`)
- Optional: NVIDIA GPU with proper drivers for CUDA acceleration


## Installation

Install from the project root (editable dev install):

```bash
python -m pip install -e .
```

If you use a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -e .
```


## Quickstart

- Download audio to `data/raw/audio`:

```bash
ytpipe download -s "https://www.youtube.com/watch?v=..." --out data/raw
```

- Transcribe everything from `data/raw/audio` to `data/transcripts`:

```bash
ytpipe transcribe --audio-dir data/raw/audio --out data/transcripts --model medium
```

- Run both steps in sequence:

```bash
ytpipe pipeline -s "https://www.youtube.com/watch?v=..." --out data
```

Resulting layout (example):

```
data/
  raw/
    audio/
      2FkMfgNNlZ8.m4a
      2FkMfgNNlZ8.info.json
      ...
    manifest.jsonl
    downloaded.txt
  transcripts/
    2FkMfgNNlZ8.json
    2FkMfgNNlZ8.txt
```


## CLI Reference

### download

```bash
ytpipe download \
  --source <url> [-s <url> ...] \
  --out data/raw \
  [--max <N>] \
  [--no-archive] \
  [--progress/--no-progress] \
  [--ffmpeg <path-to-ffmpeg-dir>]
```

- **--source, -s**: YouTube video/playlist/channel URL. Repeat to add multiple sources.
- **--out**: Output base directory for downloads. Audio is saved under `audio/` subfolder.
- **--max**: Optional limit on total videos processed.
- **--no-archive**: Don’t use download archive (forces re-downloads). Default uses `downloaded.txt`.
- **--progress/--no-progress**: Toggle tqdm progress bar.
- **--ffmpeg**: Path to a directory containing `ffmpeg` (and `ffprobe`) binaries.

Writes:
- `data/raw/audio/<id>.m4a`
- `data/raw/audio/<id>.info.json` (per-item metadata from yt-dlp)
- `data/raw/manifest.jsonl` (one JSON object per encountered item)
- `data/raw/downloaded.txt` (yt-dlp download archive, if enabled)


### transcribe

```bash
ytpipe transcribe \
  --audio-dir data/raw/audio \
  --out data/transcripts \
  [--model small|medium|large-v3|<path>] \
  [--device auto|cuda|cpu] \
  [--compute-type auto|float16|int8] \
  [--beam-size 5] \
  [--vad/--no-vad] \
  [--language <code>] \
  [--skip-existing/--no-skip-existing]
```

- **--audio-dir**: Directory containing audio files to transcribe.
- **--out**: Where transcript files are written.
- **--model**: Faster‑Whisper model name or local path (e.g. `small`, `medium`, `large-v3`).
- **--device**: `auto` picks `cuda` when available; otherwise `cpu`.
- **--compute-type**: `auto` uses `float16` on CUDA or `int8` on CPU.
- **--beam-size**: Beam search size.
- **--vad**: Enable voice activity detection filter.
- **--language**: Language hint (e.g., `en`); leave unset to auto-detect.
- **--skip-existing**: Skip files with both `.json` and `.txt` already present.

Writes, per audio `<id>`:
- `data/transcripts/<id>.json` (metadata + segments)
- `data/transcripts/<id>.txt` (plain concatenated text)

Transcript JSON shape (simplified):

```json
{
  "video_id": "2FkMfgNNlZ8",
  "source_url": "https://www.youtube.com/watch?v=2FkMfgNNlZ8",
  "model": "medium",
  "device": "cpu",
  "compute_type": "int8",
  "duration": 123.45,
  "language": "en",
  "segments": [
    { "start": 0.0, "end": 2.34, "text": "Hello world" }
  ]
}
```


### pipeline

```bash
ytpipe pipeline \
  --source <url> [-s <url> ...] \
  --out data \
  [--max <N>] [--no-archive] [--progress/--no-progress] [--ffmpeg <dir>] \
  [--model <name|path>] [--device <auto|cuda|cpu>] [--compute-type <...>] \
  [--beam-size <int>] [--vad/--no-vad] [--language <code>] \
  [--skip-existing/--no-skip-existing]
```

Convenience command that downloads to `<out>/raw` and transcribes to `<out>/transcripts`.


## Python API

You can use the same pipeline programmatically:

```python
from pathlib import Path
from ytpipe import (
    DownloadConfig,
    TranscriptionConfig,
    download_sources,
    transcribe_directory,
    setup_logging,
)

setup_logging(verbose=True)

# 1) Download
raw_dir = Path("data/raw")
items = download_sources(
    sources=["https://www.youtube.com/playlist?list=PL123..."],
    config=DownloadConfig(
        out_dir=raw_dir,
        max_videos=10,
        use_archive=True,
        show_progress=True,
    ),
)

# 2) Transcribe
results = transcribe_directory(
    audio_dir=raw_dir / "audio",
    out_dir=Path("data/transcripts"),
    config=TranscriptionConfig(
        model="medium",
        device="auto",
        compute_type="auto",
        beam_size=5,
        vad_filter=True,
        language="en",
        skip_existing=True,
    ),
)
```


## Notes and Tips

- For faster transcriptions, use an NVIDIA GPU (`--device cuda`) and pick an appropriate `--model`/`--compute-type` (e.g., `float16`).
- On CPU-only systems, `--compute-type int8` (default via `auto`) significantly reduces memory usage.
- You can limit the number of processed items with `--max` during `download`/`pipeline` runs.


## Troubleshooting

- **ffmpeg not found**:
  - Install ffmpeg and ensure it is on your `PATH`, or pass `--ffmpeg <dir>` where `ffmpeg.exe` (and `ffprobe`) live.
  - Windows (example): install from the official site or package managers (e.g., `choco install ffmpeg`), then restart your shell.

- **GPU not detected**:
  - `--device auto` checks for `nvidia-smi`. Make sure the NVIDIA driver is installed and `nvidia-smi` works in your shell. Otherwise it falls back to `cpu`.

- **Rate limiting / restricted videos**:
  - Consider fewer parallel downloads or re-running later. For age/region-restricted content you may need authentication/cookies. The CLI does not expose cookie options; from Python you can pass extra yt-dlp options via `DownloadConfig.ydl_extra_opts`.


## Development

- Install dev tools:

```bash
python -m pip install -e ".[dev]"
```

- Recommended:
  - Lint: `ruff`
  - Format: `black`
  - Types: `mypy`
  - Tests: `pytest`


## License

MIT License. See the license metadata in `pyproject.toml`.


