from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List, Optional

import typer

from .config import DownloadConfig, TranscriptionConfig, setup_logging
from .download import download_sources

app = typer.Typer(
    help="YouTube audio downloader + faster-whisper transcription pipeline."
)


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose (DEBUG) logging.",
    )
) -> None:
    """
    Common options shared by all commands.
    """
    if not shutil.which("ffmpeg"):
        typer.echo("CRITICAL: FFmpeg nije pronađen! Molimo instalirajte ga i dodajte u PATH.", err=True)
        sys.exit(1)
    setup_logging(verbose=verbose)


@app.command()
def download(
    source: List[str] = typer.Option(
        ...,
        "--source",
        "-s",
        help="YouTube URL (video/playlist/channel). Repeat for multiple.",
    ),
    out: Path = typer.Option(
        Path("data/raw"),
        "--out",
        help="Output base directory for downloads (audio/ subdir will be created).",
    ),
    max_videos: Optional[int] = typer.Option(
        None,
        "--max",
        help="Optional cap on number of videos across all sources.",
    ),
    no_archive: bool = typer.Option(
        False,
        "--no-archive",
        help="Do not use download archive (force re-download).",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show tqdm progress bar.",
    ),
    ffmpeg: Optional[Path] = typer.Option(
        None,
        "--ffmpeg",
        help="Path to ffmpeg/ffprobe binary directory (optional).",
    ),
) -> None:
    """
    Download audio from YouTube sources into m4a files + manifest.
    """
    cfg = DownloadConfig(
        out_dir=out,
        max_videos=max_videos,
        use_archive=not no_archive,
        show_progress=progress,
        ffmpeg_path=ffmpeg,
    )

    items = download_sources(source, cfg)
    typer.echo(f"Processed {len(items)} item(s).")


def _parse_output_formats(formats_str: str) -> set[str]:
    """Parse comma-separated output formats string."""
    from .config import SUPPORTED_OUTPUT_FORMATS
    
    formats = {f.strip().lower() for f in formats_str.split(",")}
    invalid = formats - SUPPORTED_OUTPUT_FORMATS
    if invalid:
        raise typer.BadParameter(
            f"Invalid format(s): {', '.join(invalid)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
        )
    return formats


@app.command()
def transcribe(
    audio_dir: Path = typer.Option(
        Path("data/raw/audio"),
        "--audio-dir",
        help="Directory containing audio files to transcribe.",
    ),
    out: Path = typer.Option(
        Path("data/transcripts"),
        "--out",
        help="Directory where transcripts will be written.",
    ),
    model: str = typer.Option(
        "medium",
        "--model",
        help="Whisper model size or path (e.g. small, medium, large-v3).",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help='Device selection: "auto", "cuda", or "cpu". Default: cpu.',
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help='Compute type (e.g. "float16" for CUDA, "int8" for CPU, or "auto").',
    ),
    beam_size: int = typer.Option(
        5,
        "--beam-size",
        help="Beam size for decoding.",
    ),
    vad: bool = typer.Option(
        False,
        "--vad/--no-vad",
        help="Enable VAD filter.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Language hint like 'en'. Leave empty to auto-detect.",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip audio files that already have all requested outputs.",
    ),
    output_format: str = typer.Option(
        "json,txt",
        "--output-format",
        "-f",
        help="Output formats (comma-separated): json, txt, srt, vtt. Example: --output-format srt,vtt",
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers. Auto-detects optimal value if not specified. Use 1 for sequential processing.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Number of files to process in parallel (per worker).",
    ),
) -> None:
    """
    Transcribe audio files in a directory using faster-whisper.

    Supports parallel processing for improved performance:
    - Sequential (default): --workers 1
    - Parallel CPU: --workers 4 (or more) with --device cpu
    - Multi-GPU: --workers 2 (or more) with --device cuda (auto-detects GPUs)

    Examples:
        # Sequential processing (default)
        ytpipe transcribe --audio-dir data/audio

        # Parallel CPU processing with 4 workers
        ytpipe transcribe --audio-dir data/audio --device cpu --workers 4

        # Multi-GPU processing (auto-detects available GPUs)
        ytpipe transcribe --audio-dir data/audio --device cuda --workers 2
    """
    from .transcribe import transcribe_directory

    formats = _parse_output_formats(output_format)

    if num_workers is None:
        import multiprocessing as mp
        if device == "cpu":
            num_workers = max(1, (mp.cpu_count() or 1) - 1)
        else:
            num_workers = 1

    cfg = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad,
        language=language,
        skip_existing=skip_existing,
        output_formats=formats,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    results = transcribe_directory(audio_dir=audio_dir, out_dir=out, config=cfg)
    typer.echo(f"Transcribed {len(results)} new file(s).")


@app.command()
def pipeline(
    source: List[str] = typer.Option(
        ...,
        "--source",
        "-s",
        help="YouTube URL (video/playlist/channel). Repeat for multiple.",
    ),
    out: Path = typer.Option(
        Path("data"),
        "--out",
        help=(
            "Base data directory. "
            "Downloads go to <out>/raw, transcripts to <out>/transcripts."
        ),
    ),
    max_videos: Optional[int] = typer.Option(
        None,
        "--max",
        help="Optional cap on number of videos across all sources.",
    ),
    no_archive: bool = typer.Option(
        False,
        "--no-archive",
        help="Do not use download archive (force re-download).",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show tqdm progress bar during download.",
    ),
    ffmpeg: Optional[Path] = typer.Option(
        None,
        "--ffmpeg",
        help="Path to ffmpeg/ffprobe binary directory (optional).",
    ),
    model: str = typer.Option(
        "medium",
        "--model",
        help="Whisper model size or path (e.g. small, medium, large-v3).",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help='Device selection: "auto", "cuda", or "cpu". Default: cpu.',
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help='Compute type (e.g. "float16" for CUDA, "int8" for CPU, or "auto").',
    ),
    beam_size: int = typer.Option(
        5,
        "--beam-size",
        help="Beam size for decoding.",
    ),
    vad: bool = typer.Option(
        False,
        "--vad/--no-vad",
        help="Enable VAD filter.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        help="Language hint like 'en'. Leave empty to auto-detect.",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip audio files that already have all requested outputs.",
    ),
    output_format: str = typer.Option(
        "json,txt",
        "--output-format",
        "-f",
        help="Output formats (comma-separated): json, txt, srt, vtt. Example: --output-format srt,vtt",
    ),
    num_workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers for transcription. Auto-detects optimal value if not specified.",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        "-b",
        help="Number of files to process in parallel (per worker).",
    ),
) -> None:
    """
    Convenience command: download then transcribe in one go.

    Supports parallel transcription for improved performance.
    See 'ytpipe transcribe --help' for parallelization examples.
    """
    from .transcribe import transcribe_directory

    formats = _parse_output_formats(output_format)

    raw_dir = out / "raw"
    transcripts_dir = out / "transcripts"

    download_cfg = DownloadConfig(
        out_dir=raw_dir,
        max_videos=max_videos,
        use_archive=not no_archive,
        show_progress=progress,
        ffmpeg_path=ffmpeg,
    )

    items = download_sources(source, download_cfg)
    typer.echo(f"Downloaded/registered {len(items)} item(s).")

    if num_workers is None:
        import multiprocessing as mp
        if device == "cpu":
            num_workers = max(1, (mp.cpu_count() or 1) - 1)
        else:
            num_workers = 1

    audio_dir = raw_dir / "audio"
    trans_cfg = TranscriptionConfig(
        model=model,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        vad_filter=vad,
        language=language,
        skip_existing=skip_existing,
        output_formats=formats,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    results = transcribe_directory(
        audio_dir=audio_dir,
        out_dir=transcripts_dir,
        config=trans_cfg,
    )
    typer.echo(f"Transcribed {len(results)} new file(s).")
