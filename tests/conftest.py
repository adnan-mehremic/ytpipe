"""
Shared pytest fixtures for ytpipe test suite.

This module provides reusable fixtures following the DRY principle,
enabling consistent test setup across all test modules.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# PATH FIXTURES
# =============================================================================


@pytest.fixture
def temp_audio_dir(tmp_path: Path) -> Path:
    """Create a temporary audio directory structure."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    return audio_dir


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for transcripts."""
    out_dir = tmp_path / "transcripts"
    out_dir.mkdir(parents=True)
    return out_dir


@pytest.fixture
def temp_raw_dir(tmp_path: Path) -> Path:
    """Create a temporary raw data directory (for downloads)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    return raw_dir


# =============================================================================
# SAMPLE DATA FIXTURES
# =============================================================================


@pytest.fixture
def sample_video_id() -> str:
    """Return a sample YouTube video ID."""
    return "dQw4w9WgXcQ"


@pytest.fixture
def sample_video_entry(sample_video_id: str) -> Dict[str, Any]:
    """
    Return a sample yt-dlp video entry dict.
    
    This mimics the structure returned by yt-dlp's extract_info().
    """
    return {
        "id": sample_video_id,
        "title": "Sample Video Title",
        "uploader": "Sample Channel",
        "webpage_url": f"https://www.youtube.com/watch?v={sample_video_id}",
        "duration": 212.0,
        "upload_date": "20231015",
        "ext": "m4a",
    }


@pytest.fixture
def sample_playlist_entry(sample_video_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a sample yt-dlp playlist entry with nested videos.
    
    Tests the flattening logic for playlists/channels.
    """
    video2 = sample_video_entry.copy()
    video2["id"] = "abc123XYZ"
    video2["title"] = "Second Video"
    
    return {
        "id": "PLtest123",
        "title": "Test Playlist",
        "_type": "playlist",
        "entries": [sample_video_entry, video2],
    }


@pytest.fixture
def sample_nested_playlist_entry(sample_playlist_entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a deeply nested playlist structure (channel with playlists).
    
    Tests recursive flattening.
    """
    return {
        "id": "UCtest123",
        "title": "Test Channel",
        "_type": "channel",
        "entries": [
            sample_playlist_entry,
            {
                "id": "singleVideo123",
                "title": "Standalone Video",
                "uploader": "Channel",
                "webpage_url": "https://www.youtube.com/watch?v=singleVideo123",
                "duration": 300.0,
                "upload_date": "20231020",
            },
        ],
    }


@pytest.fixture
def sample_info_json(sample_video_id: str, temp_audio_dir: Path) -> Path:
    """Create a sample .info.json file and return its path."""
    info_path = temp_audio_dir / f"{sample_video_id}.info.json"
    info_data = {
        "id": sample_video_id,
        "title": "Sample Video",
        "uploader": "Test Channel",
        "duration": 180,
        "upload_date": "20231001",
    }
    info_path.write_text(json.dumps(info_data), encoding="utf-8")
    return info_path


# =============================================================================
# AUDIO FILE FIXTURES
# =============================================================================


@pytest.fixture
def sample_audio_files(temp_audio_dir: Path) -> List[Path]:
    """
    Create sample audio files with various extensions.
    
    Returns list of created file paths for verification.
    """
    extensions = [".m4a", ".mp3", ".wav", ".flac", ".webm"]
    files = []
    
    for i, ext in enumerate(extensions):
        audio_file = temp_audio_dir / f"video_{i:03d}{ext}"
        audio_file.write_bytes(b"\x00" * 1024)  # Dummy content
        files.append(audio_file)
    
    return files


@pytest.fixture
def sample_mixed_files(temp_audio_dir: Path) -> Dict[str, List[Path]]:
    """
    Create a mix of audio and non-audio files.
    
    Returns dict with 'audio' and 'other' file lists.
    """
    audio_files = []
    other_files = []
    
    # Audio files
    for ext in [".m4a", ".mp3", ".wav"]:
        f = temp_audio_dir / f"audio{ext}"
        f.write_bytes(b"\x00" * 512)
        audio_files.append(f)
    
    # Non-audio files (should be ignored)
    for ext in [".txt", ".json", ".py", ".jpg"]:
        f = temp_audio_dir / f"other{ext}"
        f.write_text("content", encoding="utf-8")
        other_files.append(f)
    
    return {"audio": audio_files, "other": other_files}


# =============================================================================
# MOCK FIXTURES
# =============================================================================


@pytest.fixture
def mock_nvidia_smi_available() -> Generator[MagicMock, None, None]:
    """
    Mock subprocess.run to simulate nvidia-smi being available (CUDA present).
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        yield mock_run


@pytest.fixture
def mock_nvidia_smi_unavailable() -> Generator[MagicMock, None, None]:
    """
    Mock subprocess.run to simulate nvidia-smi not found (CPU only).
    """
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
        yield mock_run


@pytest.fixture
def mock_nvidia_smi_error() -> Generator[MagicMock, None, None]:
    """
    Mock subprocess.run to simulate nvidia-smi failing (driver issue).
    """
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1)
        yield mock_run


@pytest.fixture
def mock_whisper_model() -> MagicMock:
    """
    Create a mock WhisperModel for testing transcription without actual model.
    """
    mock_model = MagicMock()
    mock_model.model_path = "medium"
    
    # Mock transcribe() to return realistic segment structure
    mock_segment = MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 5.0
    mock_segment.text = " Hello, this is a test transcription."
    
    mock_segment2 = MagicMock()
    mock_segment2.start = 5.0
    mock_segment2.end = 10.0
    mock_segment2.text = " This is the second segment."
    
    mock_info = MagicMock()
    mock_info.language = "en"
    mock_info.duration = 10.0
    
    mock_model.transcribe.return_value = (iter([mock_segment, mock_segment2]), mock_info)
    
    return mock_model


@pytest.fixture
def mock_yt_dlp() -> Generator[MagicMock, None, None]:
    """
    Mock yt_dlp.YoutubeDL for testing download logic without network calls.
    """
    with patch("yt_dlp.YoutubeDL") as mock_ydl_class:
        mock_ydl = MagicMock()
        mock_ydl_class.return_value.__enter__ = MagicMock(return_value=mock_ydl)
        mock_ydl_class.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_ydl


# =============================================================================
# TRANSCRIPT FIXTURES
# =============================================================================


@pytest.fixture
def existing_transcript_files(
    temp_output_dir: Path, sample_video_id: str
) -> Dict[str, Path]:
    """
    Create existing transcript files to test skip_existing logic.
    """
    json_path = temp_output_dir / f"{sample_video_id}.json"
    txt_path = temp_output_dir / f"{sample_video_id}.txt"
    
    json_path.write_text(
        json.dumps({"video_id": sample_video_id, "segments": []}),
        encoding="utf-8",
    )
    txt_path.write_text("Existing transcript content\n", encoding="utf-8")
    
    return {"json": json_path, "txt": txt_path}


# =============================================================================
# CONFIG FIXTURES
# =============================================================================


@pytest.fixture
def download_config(temp_raw_dir: Path):
    """Create a DownloadConfig with temp directory."""
    from ytpipe.config import DownloadConfig
    
    return DownloadConfig(
        out_dir=temp_raw_dir,
        max_videos=5,
        use_archive=True,
        show_progress=False,
    )


@pytest.fixture
def transcription_config():
    """Create a TranscriptionConfig with test-friendly settings."""
    from ytpipe.config import TranscriptionConfig, DEFAULT_OUTPUT_FORMATS
    
    return TranscriptionConfig(
        model="tiny",  # Smallest model for fast tests
        device="cpu",
        compute_type="int8",
        beam_size=1,
        vad_filter=False,
        language="en",
        skip_existing=True,
        output_formats=set(DEFAULT_OUTPUT_FORMATS),  # Explicitly set default formats
    )

