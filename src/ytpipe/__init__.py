from __future__ import annotations

from .config import (
    DownloadConfig,
    TranscriptionConfig,
    setup_logging,
    SUPPORTED_OUTPUT_FORMATS,
    DEFAULT_OUTPUT_FORMATS,
)
from .download import DownloadedItem, download_sources
from .transcribe import (
    Segment,
    TranscriptionResult,
    transcribe_directory,
    resolve_device_and_compute,
    segments_to_srt,
    segments_to_vtt,
)

__all__ = [
    "DownloadConfig",
    "TranscriptionConfig",
    "DownloadedItem",
    "download_sources",
    "Segment",
    "TranscriptionResult",
    "transcribe_directory",
    "resolve_device_and_compute",
    "segments_to_srt",
    "segments_to_vtt",
    "setup_logging",
    "SUPPORTED_OUTPUT_FORMATS",
    "DEFAULT_OUTPUT_FORMATS",
]

__version__ = "0.1.0"
