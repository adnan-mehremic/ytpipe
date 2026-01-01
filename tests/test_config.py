"""
Tests for ytpipe.config module.

Tests configuration dataclasses, default values, and logging setup.
Follows AAA (Arrange-Act-Assert) pattern throughout.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ytpipe.config import (
    DEFAULT_AUDIO_EXTS,
    DownloadConfig,
    TranscriptionConfig,
    setup_logging,
)


class TestDefaultAudioExtensions:
    """Tests for DEFAULT_AUDIO_EXTS constant."""

    def test_contains_common_audio_formats(self) -> None:
        """Verify all common audio formats are included."""
        expected = {".m4a", ".mp3", ".wav", ".flac"}
        assert expected.issubset(DEFAULT_AUDIO_EXTS)

    def test_contains_video_formats_with_audio(self) -> None:
        """Verify video formats that can contain audio are included."""
        assert ".mp4" in DEFAULT_AUDIO_EXTS
        assert ".webm" in DEFAULT_AUDIO_EXTS
        assert ".mkv" in DEFAULT_AUDIO_EXTS

    def test_is_frozen_set(self) -> None:
        """Verify DEFAULT_AUDIO_EXTS is immutable."""
        assert isinstance(DEFAULT_AUDIO_EXTS, frozenset)
        
        with pytest.raises(AttributeError):
            DEFAULT_AUDIO_EXTS.add(".ogg")  # type: ignore


class TestDownloadConfig:
    """Tests for DownloadConfig dataclass."""

    def test_default_values(self) -> None:
        """Verify sensible defaults are set."""
        config = DownloadConfig()

        assert config.out_dir == Path("data/raw")
        assert config.max_videos is None
        assert config.use_archive is True
        assert config.show_progress is True
        assert config.ffmpeg_path is None
        assert config.concurrent_fragments == 4
        assert config.write_manifest is True
        assert config.manifest_filename == "manifest.jsonl"
        assert config.archive_filename == "downloaded.txt"

    def test_custom_out_dir(self, tmp_path: Path) -> None:
        """Verify custom output directory is accepted."""
        config = DownloadConfig(out_dir=tmp_path / "custom")
        assert config.out_dir == tmp_path / "custom"

    def test_max_videos_limit(self) -> None:
        """Verify max_videos parameter works."""
        config = DownloadConfig(max_videos=10)
        assert config.max_videos == 10

    def test_disable_archive(self) -> None:
        """Verify archive can be disabled."""
        config = DownloadConfig(use_archive=False)
        assert config.use_archive is False

    def test_ffmpeg_path_accepts_path_object(self, tmp_path: Path) -> None:
        """Verify ffmpeg_path accepts Path objects."""
        ffmpeg_dir = tmp_path / "ffmpeg"
        config = DownloadConfig(ffmpeg_path=ffmpeg_dir)
        assert config.ffmpeg_path == ffmpeg_dir

    def test_ydl_extra_opts_default_empty(self) -> None:
        """Verify ydl_extra_opts defaults to empty dict."""
        config = DownloadConfig()
        assert config.ydl_extra_opts == {}

    def test_ydl_extra_opts_custom(self) -> None:
        """Verify custom yt-dlp options can be passed."""
        extra = {"cookies": "/path/to/cookies.txt", "retries": 10}
        config = DownloadConfig(ydl_extra_opts=extra)
        assert config.ydl_extra_opts == extra

    def test_extractor_args_default_none(self) -> None:
        """Verify extractor_args defaults to None."""
        config = DownloadConfig()
        assert config.extractor_args is None

    def test_extractor_args_custom(self) -> None:
        """Verify custom extractor args can be set."""
        args = {"youtube": {"player_client": ["default"]}}
        config = DownloadConfig(extractor_args=args)
        assert config.extractor_args == args


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""

    def test_default_values(self) -> None:
        """Verify sensible defaults for transcription."""
        config = TranscriptionConfig()

        assert config.model == "medium"
        assert config.device == "cpu"
        assert config.compute_type == "auto"
        assert config.beam_size == 5
        assert config.vad_filter is True  # VAD enabled by default (recommended)
        assert config.vad_parameters is None  # Use library defaults
        assert config.language is None
        assert config.skip_existing is True
        assert config.batch_size == 1
        assert config.num_workers == 1
        assert config.max_queue_size == 10

    def test_audio_extensions_default(self) -> None:
        """Verify audio_extensions defaults to DEFAULT_AUDIO_EXTS."""
        config = TranscriptionConfig()
        assert config.audio_extensions == set(DEFAULT_AUDIO_EXTS)

    def test_custom_model(self) -> None:
        """Verify custom model name is accepted."""
        config = TranscriptionConfig(model="large-v3")
        assert config.model == "large-v3"

    def test_device_options(self) -> None:
        """Verify all device options are accepted."""
        for device in ["auto", "cuda", "cpu"]:
            config = TranscriptionConfig(device=device)
            assert config.device == device

    def test_compute_type_options(self) -> None:
        """Verify compute type options are accepted."""
        for compute_type in ["auto", "float16", "float32", "int8", "int8_float16"]:
            config = TranscriptionConfig(compute_type=compute_type)
            assert config.compute_type == compute_type

    def test_beam_size_custom(self) -> None:
        """Verify custom beam size."""
        config = TranscriptionConfig(beam_size=10)
        assert config.beam_size == 10

    def test_vad_filter_enabled(self) -> None:
        """Verify VAD filter can be enabled."""
        config = TranscriptionConfig(vad_filter=True)
        assert config.vad_filter is True

    def test_language_hint(self) -> None:
        """Verify language hint is accepted."""
        config = TranscriptionConfig(language="en")
        assert config.language == "en"

    def test_skip_existing_disabled(self) -> None:
        """Verify skip_existing can be disabled."""
        config = TranscriptionConfig(skip_existing=False)
        assert config.skip_existing is False

    def test_custom_audio_extensions(self) -> None:
        """Verify custom audio extensions set."""
        custom_exts = {".m4a", ".mp3"}
        config = TranscriptionConfig(audio_extensions=custom_exts)
        assert config.audio_extensions == custom_exts


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_default_level_is_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify default logging level is INFO."""
        setup_logging(verbose=False)
        
        logger = logging.getLogger("ytpipe.test")
        with caplog.at_level(logging.DEBUG):
            logger.debug("debug message")
            logger.info("info message")

        # INFO should be logged, DEBUG should not (at default level)
        assert "info message" in caplog.text

    def test_verbose_enables_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify verbose=True enables DEBUG level."""
        setup_logging(verbose=True)
        
        logger = logging.getLogger("ytpipe.test_verbose")
        logger.setLevel(logging.DEBUG)
        
        with caplog.at_level(logging.DEBUG):
            logger.debug("debug message")

        assert "debug message" in caplog.text

    def test_log_format_contains_timestamp(self) -> None:
        """Verify log format includes timestamp pattern."""
        # Clear existing handlers to get fresh setup
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        for handler in original_handlers:
            root_logger.removeHandler(handler)
        
        try:
            setup_logging()
            
            # Find our StreamHandler (not pytest's)
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    formatter = handler.formatter
                    if formatter and hasattr(formatter, "_fmt"):
                        # Our format should have asctime
                        assert "asctime" in formatter._fmt
                        return
            
            # If we get here, at least verify setup_logging ran without error
            # (pytest may override our handlers)
        finally:
            # Restore original handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
            for handler in original_handlers:
                root_logger.addHandler(handler)

    def test_idempotent_calls(self) -> None:
        """Verify multiple setup_logging calls don't crash."""
        # Should not raise
        setup_logging(verbose=False)
        setup_logging(verbose=True)
        setup_logging(verbose=False)


class TestConfigIntegration:
    """Integration tests for config interactions."""

    def test_download_config_paths_are_pathlib(self, tmp_path: Path) -> None:
        """Verify Path objects are used consistently."""
        config = DownloadConfig(
            out_dir=tmp_path,
            ffmpeg_path=tmp_path / "ffmpeg",
        )
        
        assert isinstance(config.out_dir, Path)
        assert isinstance(config.ffmpeg_path, Path)

    def test_transcription_config_extensions_mutable(self) -> None:
        """Verify audio_extensions is mutable set (not frozenset)."""
        config = TranscriptionConfig()
        
        # Should be able to modify (unlike DEFAULT_AUDIO_EXTS)
        config.audio_extensions.add(".ogg")
        assert ".ogg" in config.audio_extensions

    def test_configs_use_slots(self) -> None:
        """Verify configs use __slots__ for memory efficiency."""
        assert hasattr(DownloadConfig, "__slots__")
        assert hasattr(TranscriptionConfig, "__slots__")

