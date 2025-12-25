from __future__ import annotations

from .config import DownloadConfig, TranscriptionConfig, setup_logging
from .download import DownloadedItem, download_sources
from .transcribe import (
    Segment,
    TranscriptionResult,
    transcribe_directory,
    resolve_device_and_compute,
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
    "setup_logging",
]

__version__ = "0.1.0"
