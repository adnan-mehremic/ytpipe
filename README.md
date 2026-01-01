# ytpipe

[![PyPI version](https://badge.fury.io/py/ytpipe.svg)](https://badge.fury.io/py/ytpipe)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

YouTube audio download and transcription pipeline.

**ytpipe** downloads audio from YouTube (videos, playlists, channels) and transcribes locally using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Designed for practical, repeatable ingestion of YouTube content into clean on-disk datasets.

## Features

- **One-command pipeline**: download + transcribe in one step
- **Robust downloads** powered by [yt-dlp](https://github.com/yt-dlp/yt-dlp) (videos, playlists, channels)
- **Fast local transcription** via faster-whisper (CPU or NVIDIA GPU)
- **VAD enabled by default** for better quality and faster processing
- **Multiple output formats**: JSON, TXT, SRT, VTT subtitles
- **Parallel processing** for batch transcription (multi-CPU and multi-GPU)

## Requirements

- Python >= 3.10
- FFmpeg (must be in PATH or provided via `--ffmpeg`)
- Optional: NVIDIA GPU + CUDA for acceleration

## Installation

### CPU only (default)

```bash
pip install ytpipe
```

### GPU support (NVIDIA CUDA)

```bash
pip install "ytpipe[gpu]"
```

## Quick Start

### Full pipeline (download + transcribe)

```bash
# CPU (default)
ytpipe pipeline -s "https://www.youtube.com/watch?v=VIDEO_ID" --out data

# GPU accelerated
ytpipe pipeline -s "https://www.youtube.com/watch?v=VIDEO_ID" --out data --device cuda
```

### Download only

```bash
ytpipe download -s "https://www.youtube.com/watch?v=VIDEO_ID" --out data/raw
```

### Download from channel (with limit)

```bash
# Download 5 videos from a channel
ytpipe pipeline -s "https://www.youtube.com/@ChannelName" --out data --max 5

# Or download only (without transcription)
ytpipe download -s "https://www.youtube.com/@ChannelName" --out data/raw --max 10
```

### Transcribe only

```bash
ytpipe transcribe --audio-dir data/raw/audio --out data/transcripts
```

## Output Structure

```
data/
  raw/
    audio/
      VIDEO_ID.m4a
      VIDEO_ID.info.json
    manifest.jsonl
    downloaded.txt
  transcripts/
    VIDEO_ID.json
    VIDEO_ID.txt
```

## CLI Options

### Common options

| Option | Description |
|--------|-------------|
| `--model` | Whisper model: `small`, `medium`, `large-v3` (default: `medium`) |
| `--device` | Device: `cpu`, `cuda`, `auto` (default: `cpu`) |
| `--language` | Language hint (e.g., `en`). Auto-detects if not set |
| `--output-format` | Output formats: `json,txt,srt,vtt` (default: `json,txt`) |

### GPU acceleration

```bash
ytpipe transcribe --audio-dir data/raw/audio --device cuda --compute-type float16
```

### Parallel processing

```bash
# CPU parallel (4 workers)
ytpipe transcribe --audio-dir data/raw/audio --workers 4

# Multi-GPU
ytpipe transcribe --audio-dir data/raw/audio --device cuda --workers 2
```

## Python API

```python
from pathlib import Path
from ytpipe import (
    DownloadConfig,
    TranscriptionConfig,
    download_sources,
    transcribe_directory,
)

# Download
items = download_sources(
    sources=["https://www.youtube.com/watch?v=VIDEO_ID"],
    config=DownloadConfig(out_dir=Path("data/raw")),
)

# Transcribe
results = transcribe_directory(
    audio_dir=Path("data/raw/audio"),
    out_dir=Path("data/transcripts"),
    config=TranscriptionConfig(model="medium"),
)
```

## Troubleshooting

- **FFmpeg not found**: Install FFmpeg and add to PATH, or use `--ffmpeg /path/to/ffmpeg`
- **GPU not detected**: Ensure NVIDIA drivers are installed and `nvidia-smi` works
- **DLL errors on Windows**: Use `--device cpu` (default) to avoid CUDA dependencies

## License

MIT License
