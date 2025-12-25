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

# Supported output formats for transcription
SUPPORTED_OUTPUT_FORMATS: FrozenSet[str] = frozenset({"json", "txt", "srt", "vtt"})
DEFAULT_OUTPUT_FORMATS: FrozenSet[str] = frozenset({"json", "txt"})


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

    Voice Activity Detection (VAD) improves transcription quality by:
    - Filtering out silence and background noise
    - Reducing hallucinations (false transcriptions in quiet segments)
    - Improving timestamp accuracy
    - Speeding up processing by 20-30%
    """

    model: str = "medium"
    device: str = "auto"  # "auto", "cuda", or "cpu"
    compute_type: str = "auto"  # e.g., "float16", "int8", or "auto"
    beam_size: int = 5
    vad_filter: bool = True  # Enable VAD by default (recommended)
    language: str | None = None
    audio_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_AUDIO_EXTS)
    )
    skip_existing: bool = True
    output_formats: set[str] = field(
        default_factory=lambda: set(DEFAULT_OUTPUT_FORMATS)
    )  # Supported: "json", "txt", "srt", "vtt"

    # Voice Activity Detection (VAD) parameters
    # These parameters control how VAD detects speech vs silence
    vad_parameters: dict[str, Any] | None = None  # Auto-uses optimal defaults if None

    # Batch processing and parallelization options
    batch_size: int = 1  # Number of files to process in parallel
    num_workers: int = 1  # Number of parallel worker processes/threads
    max_queue_size: int = 10  # Maximum number of files to queue for processing


def get_default_vad_parameters() -> dict[str, Any]:
    """
    Get default Voice Activity Detection (VAD) parameters optimized for quality.

    These parameters provide a good balance between accuracy and speed for most
    use cases. They can be overridden via TranscriptionConfig.vad_parameters.

    Returns
    -------
    dict[str, Any]
        Dictionary of VAD parameters for faster-whisper.

    Notes
    -----
    Parameters explained:
    - threshold: Speech detection threshold (0-1). Lower = more sensitive.
    - min_speech_duration_ms: Minimum duration to consider as speech.
    - max_speech_duration_s: Maximum duration before splitting segment.
    - min_silence_duration_ms: Minimum silence to split segments.
    - window_size_samples: VAD window size (don't change unless you know what you're doing).
    - speech_pad_ms: Padding added to speech segments (helps avoid cutting words).

    Examples
    --------
    >>> params = get_default_vad_parameters()
    >>> params['threshold']
    0.5
    """
    return {
        "threshold": 0.5,  # Speech probability threshold (0-1)
        "min_speech_duration_ms": 250,  # Minimum speech duration (ms)
        "max_speech_duration_s": float("inf"),  # No max duration
        "min_silence_duration_ms": 2000,  # Minimum silence between segments (ms)
        "window_size_samples": 1024,  # VAD analysis window size
        "speech_pad_ms": 400,  # Padding around speech segments (ms)
    }


def validate_vad_parameters(params: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize VAD parameters.

    Ensures all parameters are within valid ranges and have correct types.
    Missing parameters are filled with defaults.

    Parameters
    ----------
    params : dict[str, Any]
        User-provided VAD parameters (may be incomplete).

    Returns
    -------
    dict[str, Any]
        Validated and complete VAD parameters.

    Raises
    ------
    ValueError
        If any parameter is out of valid range or has wrong type.

    Examples
    --------
    >>> params = {"threshold": 0.3}
    >>> validated = validate_vad_parameters(params)
    >>> validated['threshold']
    0.3
    >>> 'min_speech_duration_ms' in validated
    True
    """
    # Start with defaults
    validated = get_default_vad_parameters()

    # Update with user-provided values
    if params:
        validated.update(params)

    # Validate threshold (0-1)
    if not 0 <= validated["threshold"] <= 1:
        raise ValueError(
            f"VAD threshold must be between 0 and 1, got {validated['threshold']}"
        )

    # Validate min_speech_duration_ms (positive)
    if validated["min_speech_duration_ms"] <= 0:
        raise ValueError(
            f"min_speech_duration_ms must be positive, got {validated['min_speech_duration_ms']}"
        )

    # Validate max_speech_duration_s (positive or inf)
    if validated["max_speech_duration_s"] <= 0 and validated["max_speech_duration_s"] != float("inf"):
        raise ValueError(
            f"max_speech_duration_s must be positive or inf, got {validated['max_speech_duration_s']}"
        )

    # Validate min_silence_duration_ms (non-negative)
    if validated["min_silence_duration_ms"] < 0:
        raise ValueError(
            f"min_silence_duration_ms must be non-negative, got {validated['min_silence_duration_ms']}"
        )

    # Validate window_size_samples (positive power of 2)
    window_size = validated["window_size_samples"]
    if window_size <= 0 or (window_size & (window_size - 1)) != 0:
        raise ValueError(
            f"window_size_samples must be a positive power of 2, got {window_size}"
        )

    # Validate speech_pad_ms (non-negative)
    if validated["speech_pad_ms"] < 0:
        raise ValueError(
            f"speech_pad_ms must be non-negative, got {validated['speech_pad_ms']}"
        )

    return validated


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
