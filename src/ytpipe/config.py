from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, FrozenSet
import logging
import warnings

# Default audio extensions supported for transcription
DEFAULT_AUDIO_EXTS: FrozenSet[str] = frozenset(
    {".m4a", ".mp3", ".mp4", ".webm", ".wav", ".flac", ".mkv"}
)


@dataclass(slots=True)
class DownloadConfig:
    """
    Configuration options for YouTube audio downloads.
    """

    out_dir: Path = Path("data/raw")
    max_videos: int | None = None
    use_archive: bool = True
    show_progress: bool = True
    ffmpeg_path: Path | None = None
    concurrent_fragments: int = 4
    write_manifest: bool = True
    manifest_filename: str = "manifest.jsonl"
    archive_filename: str = "downloaded.txt"

    # By default, apply yt-dlp extractor_args workaround to silence
    # "No supported JavaScript runtime" warning by forcing player_client=default.
    # You can override or disable this by setting extractor_args=None when
    # constructing DownloadConfig.
    extractor_args: dict[str, Any] | None = None

    # Arbitrary extra yt-dlp options, merged into ydl_opts at the end.
    ydl_extra_opts: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TranscriptionConfig:
    """
    Configuration options for transcription with faster-whisper.
    """

    model: str = "medium"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    compute_type: str = "auto"  # e.g., "float16", "int8", or "auto"
    beam_size: int = 5
    vad_filter: bool = False
    language: str | None = None
    audio_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_AUDIO_EXTS)
    )
    skip_existing: bool = True


def setup_logging(verbose: bool = False) -> None:
    """
    Initialize the root logger with a standard format and filter noisy
    pkg_resources deprecation warnings from ctranslate2/faster-whisper.

    Parameters
    ----------
    verbose:
        If True, set log level to DEBUG; otherwise INFO.
    """
    # Filter the specific pkg_resources deprecation warning spam.
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"pkg_resources is deprecated as an API.*",
    )

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
