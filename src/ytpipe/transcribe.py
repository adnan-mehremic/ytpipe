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
    json_path: Path
    txt_path: Path
    language: str | None
    duration: float | None
    segments: list[Segment]


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
        Directory where transcript outputs (json/txt) will be written.
    config:
        Transcription configuration.
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
    json_path = out_dir / f"{stem}.json"
    txt_path = out_dir / f"{stem}.txt"

    if (
        config.skip_existing
        and json_path.exists()
        and txt_path.exists()
    ):
        logger.info("Skipping %s (already transcribed).", stem)
        return None

    logger.info("Transcribing: %s", audio_path.name)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=config.language,
        vad_filter=config.vad_filter,
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

    out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(meta, f_json, ensure_ascii=False)
    with txt_path.open("w", encoding="utf-8") as f_txt:
        f_txt.write("\n".join(texts) + "\n")

    logger.info("Wrote %s and %s", json_path.name, txt_path.name)

    return TranscriptionResult(
        audio_path=audio_path,
        json_path=json_path,
        txt_path=txt_path,
        language=meta["language"],
        duration=meta["duration"],
        segments=segments,
    )


def transcribe_directory(
    audio_dir: Path,
    out_dir: Path,
    config: TranscriptionConfig,
) -> list[TranscriptionResult]:
    """
    Transcribe all supported audio files in a directory using faster-whisper.

    Parameters
    ----------
    audio_dir:
        Directory containing audio files (e.g., data/raw/audio).
    out_dir:
        Directory where transcription outputs will be written.
    config:
        Transcription configuration.

    Returns
    -------
    list[TranscriptionResult]
        List of results for newly transcribed files.
    """
    # Lazy import so that plain "ytpipe download" does not pull in faster-whisper
    # and ctranslate2 (which triggers pkg_resources warnings).
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    out_dir.mkdir(parents=True, exist_ok=True)

    device, compute_type = resolve_device_and_compute(
        config.device, config.compute_type
    )
    logger.info(
        "Loading faster-whisper model=%s | device=%s | compute_type=%s",
        config.model,
        device,
        compute_type,
    )

    model = WhisperModel(config.model, device=device, compute_type=compute_type)

    audio_files = iter_audio_files(audio_dir, config.audio_extensions)
    if not audio_files:
        logger.warning("No audio files found in %s", audio_dir)
        return []

    results: list[TranscriptionResult] = []
    for audio_path in audio_files:
        try:
            result = transcribe_file(
                model=model,
                audio_path=audio_path,
                out_dir=out_dir,
                config=config,
                device=device,
                compute_type=compute_type,
            )
            if result is not None:
                results.append(result)
        except Exception:
            logger.exception("Failed to transcribe audio file: %s", audio_path)

    logger.info("Transcribed %d new file(s).", len(results))
    return results
