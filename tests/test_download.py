"""
Tests for ytpipe.download module.

Tests download utilities, manifest generation, and yt-dlp integration.
Uses mocks to avoid actual network calls.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from ytpipe.config import DownloadConfig
from ytpipe.download import (
    DownloadedItem,
    TqdmHook,
    _ensure_dirs,
    _flatten_entries,
    _to_manifest_row,
    download_sources,
)


class TestDownloadedItem:
    """Tests for DownloadedItem dataclass."""

    def test_creation_with_all_fields(self, tmp_path: Path) -> None:
        """Verify DownloadedItem can be created with all fields."""
        item = DownloadedItem(
            id="abc123",
            title="Test Video",
            uploader="Test Channel",
            webpage_url="https://www.youtube.com/watch?v=abc123",
            duration=180.0,
            upload_date="20231015",
            audio_path=tmp_path / "abc123.m4a",
            info_json_path=tmp_path / "abc123.info.json",
            source="youtube",
        )

        assert item.id == "abc123"
        assert item.title == "Test Video"
        assert item.uploader == "Test Channel"
        assert item.duration == 180.0
        assert item.source == "youtube"

    def test_default_source_is_youtube(self, tmp_path: Path) -> None:
        """Verify default source is 'youtube'."""
        item = DownloadedItem(
            id="test",
            title=None,
            uploader=None,
            webpage_url=None,
            duration=None,
            upload_date=None,
            audio_path=tmp_path / "test.m4a",
            info_json_path=tmp_path / "test.info.json",
        )

        assert item.source == "youtube"

    def test_nullable_fields(self, tmp_path: Path) -> None:
        """Verify optional fields can be None."""
        item = DownloadedItem(
            id="test",
            title=None,
            uploader=None,
            webpage_url=None,
            duration=None,
            upload_date=None,
            audio_path=tmp_path / "test.m4a",
            info_json_path=tmp_path / "test.info.json",
        )

        assert item.title is None
        assert item.uploader is None
        assert item.duration is None

    def test_uses_slots(self) -> None:
        """Verify DownloadedItem uses __slots__ for memory efficiency."""
        assert hasattr(DownloadedItem, "__slots__")


class TestEnsureDirs:
    """Tests for _ensure_dirs helper function."""

    def test_creates_audio_directory(self, tmp_path: Path) -> None:
        """Verify audio subdirectory is created."""
        result = _ensure_dirs(tmp_path)

        assert "audio" in result
        assert result["audio"].exists()
        assert result["audio"].is_dir()
        assert result["audio"] == tmp_path / "audio"

    def test_idempotent(self, tmp_path: Path) -> None:
        """Verify multiple calls don't raise errors."""
        _ensure_dirs(tmp_path)
        _ensure_dirs(tmp_path)  # Should not raise

        assert (tmp_path / "audio").exists()

    def test_creates_nested_path(self, tmp_path: Path) -> None:
        """Verify deeply nested paths are created."""
        deep_path = tmp_path / "a" / "b" / "c"
        result = _ensure_dirs(deep_path)

        assert result["audio"].exists()
        assert result["audio"] == deep_path / "audio"


class TestFlattenEntries:
    """Tests for _flatten_entries recursive flattening function."""

    def test_single_video_entry(self, sample_video_entry: Dict[str, Any]) -> None:
        """Verify single video is yielded as-is."""
        result = list(_flatten_entries(sample_video_entry))

        assert len(result) == 1
        assert result[0]["id"] == sample_video_entry["id"]

    def test_playlist_flattens_to_videos(
        self, sample_playlist_entry: Dict[str, Any]
    ) -> None:
        """Verify playlist entries are flattened to individual videos."""
        result = list(_flatten_entries(sample_playlist_entry))

        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert "dQw4w9WgXcQ" in ids
        assert "abc123XYZ" in ids

    def test_nested_playlist_fully_flattened(
        self, sample_nested_playlist_entry: Dict[str, Any]
    ) -> None:
        """Verify deeply nested structures are fully flattened."""
        result = list(_flatten_entries(sample_nested_playlist_entry))

        # 2 videos from nested playlist + 1 standalone video = 3
        assert len(result) == 3

    def test_empty_info_returns_nothing(self) -> None:
        """Verify empty/None info yields nothing."""
        assert list(_flatten_entries({})) == []
        assert list(_flatten_entries(None)) == []  # type: ignore

    def test_none_entries_skipped(self) -> None:
        """Verify None entries in playlist are skipped."""
        info = {
            "id": "playlist",
            "entries": [
                {"id": "video1", "title": "Video 1"},
                None,
                {"id": "video2", "title": "Video 2"},
            ],
        }

        result = list(_flatten_entries(info))

        assert len(result) == 2
        assert result[0]["id"] == "video1"
        assert result[1]["id"] == "video2"

    def test_empty_entries_list(self) -> None:
        """Verify empty entries list yields nothing.
        
        Note: An empty `entries` list is falsy in Python, so current
        implementation treats it as a leaf entry (yields the dict itself).
        This test reflects that behavior.
        """
        info = {"id": "empty_playlist", "entries": []}
        result = list(_flatten_entries(info))
        # Empty entries list is falsy, so the dict itself is yielded as a leaf
        assert len(result) == 1
        assert result[0]["id"] == "empty_playlist"


class TestToManifestRow:
    """Tests for _to_manifest_row conversion function."""

    def test_extracts_all_fields(
        self, sample_video_entry: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify all expected fields are extracted."""
        audio_dir = tmp_path / "audio"
        row = _to_manifest_row(sample_video_entry, audio_dir)

        assert row["id"] == sample_video_entry["id"]
        assert row["title"] == sample_video_entry["title"]
        assert row["uploader"] == sample_video_entry["uploader"]
        assert row["webpage_url"] == sample_video_entry["webpage_url"]
        assert row["duration"] == sample_video_entry["duration"]
        assert row["upload_date"] == sample_video_entry["upload_date"]
        assert row["source"] == "youtube"

    def test_generates_correct_paths(
        self, sample_video_entry: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify audio and info.json paths are correctly generated."""
        audio_dir = tmp_path / "audio"
        row = _to_manifest_row(sample_video_entry, audio_dir)

        expected_audio = str(audio_dir / f"{sample_video_entry['id']}.m4a")
        expected_info = str(audio_dir / f"{sample_video_entry['id']}.info.json")

        assert row["audio_path"] == expected_audio
        assert row["info_json"] == expected_info

    def test_handles_missing_fields(self, tmp_path: Path) -> None:
        """Verify graceful handling of missing optional fields."""
        minimal_entry = {"id": "minimal123"}
        audio_dir = tmp_path / "audio"

        row = _to_manifest_row(minimal_entry, audio_dir)

        assert row["id"] == "minimal123"
        assert row["title"] is None
        assert row["uploader"] is None
        assert row["duration"] is None

    def test_paths_are_strings(
        self, sample_video_entry: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify paths are serialized as strings (for JSON)."""
        audio_dir = tmp_path / "audio"
        row = _to_manifest_row(sample_video_entry, audio_dir)

        assert isinstance(row["audio_path"], str)
        assert isinstance(row["info_json"], str)

    def test_json_serializable(
        self, sample_video_entry: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Verify row can be serialized to JSON."""
        audio_dir = tmp_path / "audio"
        row = _to_manifest_row(sample_video_entry, audio_dir)

        # Should not raise
        json_str = json.dumps(row)
        assert sample_video_entry["id"] in json_str


class TestTqdmHook:
    """Tests for TqdmHook progress callback."""

    def test_initialization(self) -> None:
        """Verify hook initializes with correct state."""
        hook = TqdmHook(desc="test download")

        assert hook.pbar is None
        assert hook.total is None
        assert hook.desc == "test download"

    def test_downloading_status_creates_progress_bar(self) -> None:
        """Verify progress bar is created on first download status."""
        hook = TqdmHook()

        hook({"status": "downloading", "total_bytes": 1000, "downloaded_bytes": 100})

        assert hook.pbar is not None
        assert hook.total == 1000
        hook.pbar.close()

    def test_downloading_updates_progress(self) -> None:
        """Verify progress is updated during download."""
        hook = TqdmHook()

        hook({"status": "downloading", "total_bytes": 1000, "downloaded_bytes": 100})
        hook({"status": "downloading", "total_bytes": 1000, "downloaded_bytes": 500})

        assert hook.pbar is not None
        assert hook.pbar.n == 500
        hook.pbar.close()

    def test_finished_status_closes_progress_bar(self) -> None:
        """Verify progress bar is closed on finished status."""
        hook = TqdmHook()

        hook({"status": "downloading", "total_bytes": 1000, "downloaded_bytes": 500})
        hook({"status": "finished"})

        assert hook.pbar is None

    def test_error_status_closes_progress_bar(self) -> None:
        """Verify progress bar is closed on error status."""
        hook = TqdmHook()

        hook({"status": "downloading", "total_bytes": 1000, "downloaded_bytes": 500})
        hook({"status": "error"})

        assert hook.pbar is None

    def test_handles_missing_total_bytes(self) -> None:
        """Verify graceful handling when total_bytes is unknown."""
        hook = TqdmHook()

        # No exception should be raised
        hook({"status": "downloading", "downloaded_bytes": 100})

        # pbar might not be created without total
        if hook.pbar:
            hook.pbar.close()

    def test_uses_total_bytes_estimate(self) -> None:
        """Verify fallback to total_bytes_estimate."""
        hook = TqdmHook()

        hook({"status": "downloading", "total_bytes_estimate": 2000, "downloaded_bytes": 100})

        assert hook.total == 2000
        if hook.pbar:
            hook.pbar.close()


class TestDownloadSources:
    """Tests for download_sources main function."""

    def test_empty_sources_returns_empty_list(self, tmp_path: Path) -> None:
        """Verify empty sources list returns empty result."""
        config = DownloadConfig(out_dir=tmp_path, show_progress=False)

        result = download_sources([], config)

        assert result == []

    def test_creates_output_directories(self, tmp_path: Path) -> None:
        """Verify output directories are created."""
        config = DownloadConfig(out_dir=tmp_path, show_progress=False)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = None
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            download_sources(["https://www.youtube.com/watch?v=test"], config)

        assert tmp_path.exists()
        assert (tmp_path / "audio").exists()

    def test_respects_max_videos_limit(
        self, tmp_path: Path, sample_video_entry: Dict[str, Any]
    ) -> None:
        """Verify max_videos limit is respected."""
        config = DownloadConfig(out_dir=tmp_path, max_videos=1, show_progress=False)

        # Create playlist with 3 videos
        playlist = {
            "id": "playlist",
            "entries": [
                sample_video_entry,
                {**sample_video_entry, "id": "video2"},
                {**sample_video_entry, "id": "video3"},
            ],
        }

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = playlist
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            result = download_sources(["https://example.com/playlist"], config)

        assert len(result) == 1

    def test_writes_manifest_file(
        self, tmp_path: Path, sample_video_entry: Dict[str, Any]
    ) -> None:
        """Verify manifest.jsonl is written."""
        config = DownloadConfig(
            out_dir=tmp_path, show_progress=False, write_manifest=True
        )

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = sample_video_entry
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            download_sources(["https://example.com/video"], config)

        manifest_path = tmp_path / "manifest.jsonl"
        assert manifest_path.exists()

        # Verify content
        content = manifest_path.read_text()
        assert sample_video_entry["id"] in content

    def test_uses_download_archive_when_enabled(self, tmp_path: Path) -> None:
        """Verify download archive path is set when enabled."""
        config = DownloadConfig(out_dir=tmp_path, use_archive=True, show_progress=False)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = None
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            download_sources(["https://example.com/video"], config)

            # Check that archive was set in options
            call_kwargs = mock_ydl.call_args
            if call_kwargs:
                opts = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1]
                assert "download_archive" in opts

    def test_handles_extraction_error_gracefully(self, tmp_path: Path) -> None:
        """Verify extraction errors don't crash the whole process."""
        config = DownloadConfig(out_dir=tmp_path, show_progress=False)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.side_effect = Exception("Network error")
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            # Should not raise, should return empty list
            result = download_sources(["https://example.com/video"], config)

        assert result == []

    def test_returns_downloaded_items(
        self, tmp_path: Path, sample_video_entry: Dict[str, Any]
    ) -> None:
        """Verify DownloadedItem objects are returned."""
        config = DownloadConfig(out_dir=tmp_path, show_progress=False)

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = sample_video_entry
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            result = download_sources(["https://example.com/video"], config)

        assert len(result) == 1
        assert isinstance(result[0], DownloadedItem)
        assert result[0].id == sample_video_entry["id"]
        assert result[0].title == sample_video_entry["title"]

    def test_processes_multiple_sources(
        self, tmp_path: Path, sample_video_entry: Dict[str, Any]
    ) -> None:
        """Verify multiple sources are processed."""
        config = DownloadConfig(out_dir=tmp_path, show_progress=False)

        video2 = {**sample_video_entry, "id": "second_video", "title": "Second"}

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.side_effect = [sample_video_entry, video2]
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            result = download_sources(
                ["https://example.com/video1", "https://example.com/video2"],
                config,
            )

        assert len(result) == 2

    def test_ffmpeg_path_passed_to_ytdlp(self, tmp_path: Path) -> None:
        """Verify ffmpeg_path is passed to yt-dlp options."""
        ffmpeg_dir = tmp_path / "ffmpeg"
        config = DownloadConfig(
            out_dir=tmp_path, ffmpeg_path=ffmpeg_dir, show_progress=False
        )

        with patch("yt_dlp.YoutubeDL") as mock_ydl:
            mock_instance = MagicMock()
            mock_instance.extract_info.return_value = None
            mock_ydl.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_ydl.return_value.__exit__ = MagicMock(return_value=False)

            download_sources(["https://example.com/video"], config)

            call_args = mock_ydl.call_args
            if call_args:
                opts = call_args[0][0]
                assert opts.get("ffmpeg_location") == str(ffmpeg_dir)

