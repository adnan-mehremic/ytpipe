from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Dict, List, Optional
import json
import logging
import subprocess

from .config import TranscriptionConfig, DEFAULT_AUDIO_EXTS

if TYPE_CHECKING:  # for type checkers / IDEs only
    from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Segment:
    """
    A single transcription segment.
    """

    start: float
    end: float
    text: str


@dataclass(slots=True)
class TranscriptionResult:
    """
    Result of transcribing a single audio file.
    """

    audio_path: Path
    json_path: Path | None
    txt_path: Path | None
    srt_path: Path | None
    vtt_path: Path | None
    language: str | None
    duration: float | None
    segments: list[Segment]


def _format_timestamp_srt(seconds: float) -> str:
    """
    Format seconds as SRT timestamp: HH:MM:SS,mmm
    
    Example: 3661.5 -> "01:01:01,500"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """
    Format seconds as VTT timestamp: HH:MM:SS.mmm
    
    Example: 3661.5 -> "01:01:01.500"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def segments_to_srt(segments: list[Segment]) -> str:
    """
    Convert segments to SRT subtitle format.
    
    SRT format:
        1
        00:00:00,000 --> 00:00:05,000
        First subtitle text
        
        2
        00:00:05,000 --> 00:00:10,000
        Second subtitle text
    
    Parameters
    ----------
    segments:
        List of transcription segments.
    
    Returns
    -------
    str
        SRT formatted string.
    """
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        start_ts = _format_timestamp_srt(seg.start)
        end_ts = _format_timestamp_srt(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments: list[Segment]) -> str:
    """
    Convert segments to WebVTT subtitle format.
    
    VTT format:
        WEBVTT
        
        00:00:00.000 --> 00:00:05.000
        First subtitle text
        
        00:00:05.000 --> 00:00:10.000
        Second subtitle text
    
    Parameters
    ----------
    segments:
        List of transcription segments.
    
    Returns
    -------
    str
        WebVTT formatted string.
    """
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        start_ts = _format_timestamp_vtt(seg.start)
        end_ts = _format_timestamp_vtt(seg.end)
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def resolve_device_and_compute(
    device: str, compute_type: str
) -> tuple[str, str]:
    """
    Resolve device and compute_type, with 'auto' detection.

    - If device == "auto", prefer 'cuda' when nvidia-smi is available,
      otherwise fall back to 'cpu'.
    - If compute_type == "auto", choose 'float16' for cuda, 'int8' for cpu.
    """
    resolved_device = device
    resolved_compute = compute_type

    if device == "auto":
        cuda_available = False
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            cuda_available = result.returncode == 0
        except FileNotFoundError:
            cuda_available = False

        resolved_device = "cuda" if cuda_available else "cpu"

    if compute_type == "auto":
        resolved_compute = "float16" if resolved_device == "cuda" else "int8"

    return resolved_device, resolved_compute


def iter_audio_files(
    audio_dir: Path, extensions: Collection[str] | None = None
) -> list[Path]:
    """
    Recursively find all audio files under `audio_dir` with given extensions.

    Parameters
    ----------
    audio_dir:
        Root directory to search for audio files.
    extensions:
        Allowed extensions (including dot), e.g. {".m4a", ".mp3"}.
        If None, DEFAULT_AUDIO_EXTS is used.

    Returns
    -------
    list[Path]
        Sorted list of audio file paths.
    """
    exts = {e.lower() for e in (extensions or DEFAULT_AUDIO_EXTS)}
    files: List[Path] = []

    for path in audio_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)

    return sorted(files)


def transcribe_file(
    model: "WhisperModel",  # type: ignore[name-defined]
    audio_path: Path,
    out_dir: Path,
    config: TranscriptionConfig,
    *,
    device: str,
    compute_type: str,
) -> Optional[TranscriptionResult]:
    """
    Transcribe a single audio file.

    Parameters
    ----------
    model:
        An instance of faster-whisper WhisperModel.
    audio_path:
        Path to input audio file.
    out_dir:
        Directory where transcript outputs will be written.
    config:
        Transcription configuration (includes output_formats).
    device:
        Final resolved device (e.g., "cuda" or "cpu").
    compute_type:
        Final resolved compute type.

    Returns
    -------
    Optional[TranscriptionResult]
        TranscriptionResult if a new transcription was written,
        or None if skipped (e.g., existing outputs and skip_existing=True).
    """
    stem = audio_path.stem
    formats = config.output_formats

    json_path = out_dir / f"{stem}.json" if "json" in formats else None
    txt_path = out_dir / f"{stem}.txt" if "txt" in formats else None
    srt_path = out_dir / f"{stem}.srt" if "srt" in formats else None
    vtt_path = out_dir / f"{stem}.vtt" if "vtt" in formats else None

    if config.skip_existing:
        paths_to_check = [p for p in [json_path, txt_path, srt_path, vtt_path] if p]
        if paths_to_check and all(p.exists() for p in paths_to_check):
            logger.info("Skipping %s (already transcribed).", stem)
            return None

    logger.info("Transcribing: %s", audio_path.name)

    vad_parameters = None
    if config.vad_filter:
        from .config import validate_vad_parameters
        vad_parameters = validate_vad_parameters(config.vad_parameters or {})
        logger.debug(
            "Using VAD with parameters: threshold=%.2f, min_speech=%dms, min_silence=%dms",
            vad_parameters["threshold"],
            vad_parameters["min_speech_duration_ms"],
            vad_parameters["min_silence_duration_ms"],
        )

    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.language,
        vad_filter=config.vad_filter,
        vad_parameters=vad_parameters,
        beam_size=config.beam_size,
        word_timestamps=False,
    )

    segments: list[Segment] = []
    texts: list[str] = []
    for s in segments_iter:
        seg = Segment(
            start=float(s.start),
            end=float(s.end),
            text=s.text.strip(),
        )
        segments.append(seg)
        texts.append(seg.text)

    out_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[str] = []

    if json_path:
        meta: Dict[str, Any] = {
            "video_id": stem,
            "source_url": f"https://www.youtube.com/watch?v={stem}",
            "model": getattr(model, "model_path", config.model),
            "device": device,
            "compute_type": compute_type,
            "duration": getattr(info, "duration", None),
            "language": getattr(info, "language", None),
            "segments": [asdict(seg) for seg in segments],
        }
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        written_files.append(json_path.name)

    if txt_path:
        with txt_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(texts) + "\n")
        written_files.append(txt_path.name)

    if srt_path:
        srt_content = segments_to_srt(segments)
        with srt_path.open("w", encoding="utf-8") as f:
            f.write(srt_content)
        written_files.append(srt_path.name)

    if vtt_path:
        vtt_content = segments_to_vtt(segments)
        with vtt_path.open("w", encoding="utf-8") as f:
            f.write(vtt_content)
        written_files.append(vtt_path.name)

    logger.info("Wrote %s", ", ".join(written_files))

    return TranscriptionResult(
        audio_path=audio_path,
        json_path=json_path,
        txt_path=txt_path,
        srt_path=srt_path,
        vtt_path=vtt_path,
        language=getattr(info, "language", None),
        duration=getattr(info, "duration", None),
        segments=segments,
    )


def transcribe_directory(
    audio_dir: Path,
    out_dir: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionResult]:
    """
    Transcribe all supported audio files in a directory using faster-whisper.

    This function intelligently selects the optimal processing strategy based on
    configuration and available hardware:
    - Sequential: Single-threaded processing (default, batch_size=1, num_workers=1)
    - Parallel CPU: Multi-process CPU parallelization (num_workers > 1, device=CPU)
    - Multi-GPU: Distributed processing across multiple GPUs (num_workers > 1, device=CUDA)

    Parameters
    ----------
    audio_dir:
        Directory containing audio files (e.g., data/raw/audio).
    out_dir:
        Directory where transcription outputs will be written.
    config:
        Transcription configuration including parallelization settings.

    Returns
    -------
    list[TranscriptionResult]
        List of results for newly transcribed files.

    Examples
    --------
    Sequential processing (default):
    >>> config = TranscriptionConfig(model="medium")
    >>> results = transcribe_directory(Path("audio/"), Path("output/"), config)

    Parallel CPU processing:
    >>> config = TranscriptionConfig(model="medium", num_workers=4, device="cpu")
    >>> results = transcribe_directory(Path("audio/"), Path("output/"), config)

    Multi-GPU processing:
    >>> config = TranscriptionConfig(model="medium", num_workers=2, device="cuda")
    >>> results = transcribe_directory(Path("audio/"), Path("output/"), config)

    See Also
    --------
    batch_processing.TranscriptionStrategy : Base class for processing strategies
    batch_processing.TranscriptionStrategyFactory : Factory for strategy selection
    """
    from .batch_processing import TranscriptionStrategyFactory

    out_dir.mkdir(parents=True, exist_ok=True)

    device, compute_type = resolve_device_and_compute(
        config.device, config.compute_type
    )

    audio_files = iter_audio_files(audio_dir, config.audio_extensions)
    if not audio_files:
        logger.warning("No audio files found in %s", audio_dir)
        return []

    logger.info(
        "Found %d audio file(s) to process with model=%s | device=%s | compute_type=%s",
        len(audio_files),
        config.model,
        device,
        compute_type,
    )

    strategy = TranscriptionStrategyFactory.create_strategy(config, device)
    logger.info("Using strategy: %s", strategy.get_strategy_name())

    results = strategy.process_files(
        audio_files=audio_files,
        out_dir=out_dir,
        transcribe_fn=transcribe_file,
        device=device,
        compute_type=compute_type,
    )

    logger.info("Transcribed %d new file(s).", len(results))
    return results
