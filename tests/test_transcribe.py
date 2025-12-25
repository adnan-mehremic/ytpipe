"""
Tests for ytpipe.transcribe module.

Tests transcription utilities, device detection, and file processing.
Uses mocks to avoid loading actual Whisper models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from ytpipe.config import DEFAULT_AUDIO_EXTS, TranscriptionConfig
from ytpipe.transcribe import (
    Segment,
    TranscriptionResult,
    iter_audio_files,
    resolve_device_and_compute,
    transcribe_directory,
    transcribe_file,
    segments_to_srt,
    segments_to_vtt,
    _format_timestamp_srt,
    _format_timestamp_vtt,
)


class TestSegment:
    """Tests for Segment dataclass."""

    def test_creation(self) -> None:
        """Verify Segment can be created with all fields."""
        segment = Segment(start=0.0, end=5.5, text="Hello world")

        assert segment.start == 0.0
        assert segment.end == 5.5
        assert segment.text == "Hello world"

    def test_uses_slots(self) -> None:
        """Verify Segment uses __slots__ for memory efficiency."""
        assert hasattr(Segment, "__slots__")


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_creation(self, tmp_path: Path) -> None:
        """Verify TranscriptionResult can be created with all fields."""
        segments = [Segment(0.0, 5.0, "Test")]
        result = TranscriptionResult(
            audio_path=tmp_path / "test.m4a",
            json_path=tmp_path / "test.json",
            txt_path=tmp_path / "test.txt",
            srt_path=tmp_path / "test.srt",
            vtt_path=tmp_path / "test.vtt",
            language="en",
            duration=5.0,
            segments=segments,
        )

        assert result.language == "en"
        assert result.duration == 5.0
        assert len(result.segments) == 1

    def test_nullable_fields(self, tmp_path: Path) -> None:
        """Verify optional fields can be None."""
        result = TranscriptionResult(
            audio_path=tmp_path / "test.m4a",
            json_path=tmp_path / "test.json",
            txt_path=tmp_path / "test.txt",
            srt_path=tmp_path / "test.srt",
            vtt_path=tmp_path / "test.vtt",
            language=None,
            duration=None,
            segments=[],
        )

        assert result.language is None
        assert result.duration is None

    def test_uses_slots(self) -> None:
        """Verify TranscriptionResult uses __slots__."""
        assert hasattr(TranscriptionResult, "__slots__")


class TestResolveDeviceAndCompute:
    """Tests for resolve_device_and_compute function."""

    def test_explicit_cuda_device(self) -> None:
        """Verify explicit 'cuda' device is respected."""
        device, compute = resolve_device_and_compute("cuda", "float16")

        assert device == "cuda"
        assert compute == "float16"

    def test_explicit_cpu_device(self) -> None:
        """Verify explicit 'cpu' device is respected."""
        device, compute = resolve_device_and_compute("cpu", "int8")

        assert device == "cpu"
        assert compute == "int8"

    def test_auto_device_with_cuda_available(
        self, mock_nvidia_smi_available: MagicMock
    ) -> None:
        """Verify 'auto' device resolves to 'cuda' when nvidia-smi succeeds."""
        device, compute = resolve_device_and_compute("auto", "auto")

        assert device == "cuda"
        assert compute == "float16"

    def test_auto_device_without_cuda(
        self, mock_nvidia_smi_unavailable: MagicMock
    ) -> None:
        """Verify 'auto' device resolves to 'cpu' when nvidia-smi not found."""
        device, compute = resolve_device_and_compute("auto", "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_auto_device_with_nvidia_smi_error(
        self, mock_nvidia_smi_error: MagicMock
    ) -> None:
        """Verify 'auto' device resolves to 'cpu' when nvidia-smi fails."""
        device, compute = resolve_device_and_compute("auto", "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_auto_compute_with_explicit_cuda(self) -> None:
        """Verify 'auto' compute_type resolves to 'float16' for cuda."""
        device, compute = resolve_device_and_compute("cuda", "auto")

        assert device == "cuda"
        assert compute == "float16"

    def test_auto_compute_with_explicit_cpu(self) -> None:
        """Verify 'auto' compute_type resolves to 'int8' for cpu."""
        device, compute = resolve_device_and_compute("cpu", "auto")

        assert device == "cpu"
        assert compute == "int8"

    def test_explicit_values_not_modified(self) -> None:
        """Verify explicit device and compute_type are not modified."""
        device, compute = resolve_device_and_compute("cuda", "float32")

        assert device == "cuda"
        assert compute == "float32"


class TestIterAudioFiles:
    """Tests for iter_audio_files function."""

    def test_finds_audio_files(self, sample_audio_files: List[Path]) -> None:
        """Verify audio files are found."""
        audio_dir = sample_audio_files[0].parent
        result = iter_audio_files(audio_dir)

        assert len(result) == len(sample_audio_files)

    def test_returns_sorted_paths(self, sample_audio_files: List[Path]) -> None:
        """Verify results are sorted by path."""
        audio_dir = sample_audio_files[0].parent
        result = iter_audio_files(audio_dir)

        assert result == sorted(result)

    def test_filters_by_extension(
        self, sample_mixed_files: Dict[str, List[Path]]
    ) -> None:
        """Verify non-audio files are filtered out."""
        audio_dir = sample_mixed_files["audio"][0].parent
        result = iter_audio_files(audio_dir)

        # Should only find audio files
        assert len(result) == len(sample_mixed_files["audio"])

        # Verify no non-audio files
        for path in result:
            assert path.suffix.lower() in DEFAULT_AUDIO_EXTS

    def test_custom_extensions(self, temp_audio_dir: Path) -> None:
        """Verify custom extensions filter works."""
        # Create files
        (temp_audio_dir / "test.m4a").write_bytes(b"\x00")
        (temp_audio_dir / "test.mp3").write_bytes(b"\x00")
        (temp_audio_dir / "test.wav").write_bytes(b"\x00")

        # Only look for .m4a
        result = iter_audio_files(temp_audio_dir, extensions={".m4a"})

        assert len(result) == 1
        assert result[0].suffix == ".m4a"

    def test_recursive_search(self, temp_audio_dir: Path) -> None:
        """Verify files in subdirectories are found."""
        subdir = temp_audio_dir / "subdir"
        subdir.mkdir()

        (temp_audio_dir / "root.m4a").write_bytes(b"\x00")
        (subdir / "nested.m4a").write_bytes(b"\x00")

        result = iter_audio_files(temp_audio_dir)

        assert len(result) == 2
        names = {p.name for p in result}
        assert "root.m4a" in names
        assert "nested.m4a" in names

    def test_empty_directory(self, temp_audio_dir: Path) -> None:
        """Verify empty directory returns empty list."""
        result = iter_audio_files(temp_audio_dir)
        assert result == []

    def test_case_insensitive_extensions(self, temp_audio_dir: Path) -> None:
        """Verify extension matching is case-insensitive."""
        (temp_audio_dir / "lower.m4a").write_bytes(b"\x00")
        (temp_audio_dir / "upper.M4A").write_bytes(b"\x00")
        (temp_audio_dir / "mixed.M4a").write_bytes(b"\x00")

        result = iter_audio_files(temp_audio_dir)

        assert len(result) == 3


class TestTranscribeFile:
    """Tests for transcribe_file function."""

    def test_creates_json_output(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        transcription_config: TranscriptionConfig,
    ) -> None:
        """Verify JSON transcript is created."""
        # Create dummy audio file
        audio_file = temp_audio_dir / "test_video.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        # Ensure JSON format is requested
        config = TranscriptionConfig(
            model=transcription_config.model,
            device=transcription_config.device,
            compute_type=transcription_config.compute_type,
            beam_size=transcription_config.beam_size,
            vad_filter=transcription_config.vad_filter,
            language=transcription_config.language,
            output_formats={"json"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.json_path is not None
        assert result.json_path.exists()

        # Verify JSON content
        data = json.loads(result.json_path.read_text())
        assert "video_id" in data
        assert "segments" in data
        assert data["video_id"] == "test_video"

    def test_creates_txt_output(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        transcription_config: TranscriptionConfig,
    ) -> None:
        """Verify TXT transcript is created."""
        audio_file = temp_audio_dir / "test_video.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        # Ensure TXT format is requested
        config = TranscriptionConfig(
            model=transcription_config.model,
            device=transcription_config.device,
            compute_type=transcription_config.compute_type,
            beam_size=transcription_config.beam_size,
            vad_filter=transcription_config.vad_filter,
            language=transcription_config.language,
            output_formats={"txt"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.txt_path is not None
        assert result.txt_path.exists()

        # Verify TXT content (segments joined by newlines)
        content = result.txt_path.read_text()
        assert "Hello, this is a test transcription." in content

    def test_skips_existing_when_configured(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        existing_transcript_files: Dict[str, Path],
        sample_video_id: str,
    ) -> None:
        """Verify existing transcripts are skipped when skip_existing=True."""
        audio_file = temp_audio_dir / f"{sample_video_id}.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(skip_existing=True)

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        # Should return None (skipped)
        assert result is None

        # Model should not have been called
        mock_whisper_model.transcribe.assert_not_called()

    def test_overwrites_when_skip_existing_false(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        existing_transcript_files: Dict[str, Path],
        sample_video_id: str,
    ) -> None:
        """Verify existing transcripts are overwritten when skip_existing=False."""
        audio_file = temp_audio_dir / f"{sample_video_id}.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(skip_existing=False)

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        mock_whisper_model.transcribe.assert_called_once()

    def test_returns_transcription_result(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        transcription_config: TranscriptionConfig,
    ) -> None:
        """Verify TranscriptionResult is returned with correct data."""
        audio_file = temp_audio_dir / "result_test.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=transcription_config,
            device="cpu",
            compute_type="int8",
        )

        assert isinstance(result, TranscriptionResult)
        assert result.audio_path == audio_file
        assert result.language == "en"
        assert result.duration == 10.0
        assert len(result.segments) == 2

    def test_creates_output_directory(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        tmp_path: Path,
        transcription_config: TranscriptionConfig,
    ) -> None:
        """Verify output directory is created if it doesn't exist."""
        audio_file = temp_audio_dir / "test.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        new_out_dir = tmp_path / "new" / "nested" / "dir"

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=new_out_dir,
            config=transcription_config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert new_out_dir.exists()

    def test_json_contains_metadata(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        transcription_config: TranscriptionConfig,
    ) -> None:
        """Verify JSON output contains all expected metadata."""
        audio_file = temp_audio_dir / "metadata_test.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        # Ensure JSON format is requested
        config = TranscriptionConfig(
            model=transcription_config.model,
            device=transcription_config.device,
            compute_type=transcription_config.compute_type,
            beam_size=transcription_config.beam_size,
            vad_filter=transcription_config.vad_filter,
            language=transcription_config.language,
            output_formats={"json"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.json_path is not None
        data = json.loads(result.json_path.read_text())

        assert data["video_id"] == "metadata_test"
        assert "source_url" in data
        assert data["device"] == "cpu"
        assert data["compute_type"] == "int8"
        assert "model" in data
        assert "duration" in data
        assert "language" in data
        assert "segments" in data


class TestTranscribeDirectory:
    """Tests for transcribe_directory function."""

    def test_transcribes_all_audio_files(
        self, sample_audio_files: List[Path], temp_output_dir: Path
    ) -> None:
        """Verify all audio files in directory are transcribed."""
        audio_dir = sample_audio_files[0].parent
        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            skip_existing=True,
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            # Setup mock model
            mock_model = MagicMock()
            mock_model.model_path = "tiny"  # Required for JSON serialization
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 1.0

            # Return fresh iterator each call (iterator is consumed per transcription)
            def make_transcribe_result(*args, **kwargs):
                mock_segment = MagicMock()
                mock_segment.start = 0.0
                mock_segment.end = 1.0
                mock_segment.text = "Test"
                return (iter([mock_segment]), mock_info)

            mock_model.transcribe.side_effect = make_transcribe_result
            mock_model_class.return_value = mock_model

            results = transcribe_directory(
                audio_dir=audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

        assert len(results) == len(sample_audio_files)

    def test_returns_empty_for_no_audio_files(
        self, temp_audio_dir: Path, temp_output_dir: Path
    ) -> None:
        """Verify empty list when no audio files found."""
        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )

        with patch("faster_whisper.WhisperModel"):
            results = transcribe_directory(
                audio_dir=temp_audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

        assert results == []

    def test_creates_output_directory(
        self, sample_audio_files: List[Path], tmp_path: Path
    ) -> None:
        """Verify output directory is created."""
        audio_dir = sample_audio_files[0].parent
        new_out_dir = tmp_path / "new_transcripts"
        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_path = "tiny"  # Required for JSON serialization
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 1.0

            def make_transcribe_result(*args, **kwargs):
                mock_segment = MagicMock()
                mock_segment.start = 0.0
                mock_segment.end = 1.0
                mock_segment.text = "Test"
                return (iter([mock_segment]), mock_info)

            mock_model.transcribe.side_effect = make_transcribe_result
            mock_model_class.return_value = mock_model

            transcribe_directory(
                audio_dir=audio_dir,
                out_dir=new_out_dir,
                config=config,
            )

        assert new_out_dir.exists()

    def test_handles_transcription_errors_gracefully(
        self, sample_audio_files: List[Path], temp_output_dir: Path
    ) -> None:
        """Verify errors on individual files don't stop processing."""
        audio_dir = sample_audio_files[0].parent
        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_path = "tiny"  # Required for JSON serialization
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 1.0

            # Simulate error on first file, success on rest
            def transcribe_side_effect(*args, **kwargs):
                if mock_model.transcribe.call_count == 1:
                    raise RuntimeError("Simulated transcription error")
                # Return fresh iterator each call
                mock_segment = MagicMock()
                mock_segment.start = 0.0
                mock_segment.end = 1.0
                mock_segment.text = "Test"
                return (iter([mock_segment]), mock_info)

            mock_model.transcribe.side_effect = transcribe_side_effect
            mock_model_class.return_value = mock_model

            results = transcribe_directory(
                audio_dir=audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

        # Should have processed remaining files despite first error
        assert len(results) == len(sample_audio_files) - 1

    def test_skips_existing_transcripts(
        self,
        temp_audio_dir: Path,
        temp_output_dir: Path,
        sample_video_id: str,
    ) -> None:
        """Verify existing transcripts are skipped."""
        # Create audio file
        audio_file = temp_audio_dir / f"{sample_video_id}.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        # Create existing transcript
        (temp_output_dir / f"{sample_video_id}.json").write_text("{}")
        (temp_output_dir / f"{sample_video_id}.txt").write_text("existing")

        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            skip_existing=True,
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.return_value = mock_model

            results = transcribe_directory(
                audio_dir=temp_audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

        # Should be empty since file was skipped
        assert len(results) == 0
        # Model should not have been called for transcription
        mock_model.transcribe.assert_not_called()

    def test_uses_correct_device_and_compute_type(
        self, sample_audio_files: List[Path], temp_output_dir: Path
    ) -> None:
        """Verify device and compute_type are passed to model."""
        audio_dir = sample_audio_files[0].parent
        config = TranscriptionConfig(
            model="medium",
            device="cuda",
            compute_type="float16",
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_path = "medium"  # Required for JSON serialization
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 1.0

            def make_transcribe_result(*args, **kwargs):
                mock_segment = MagicMock()
                mock_segment.start = 0.0
                mock_segment.end = 1.0
                mock_segment.text = "Test"
                return (iter([mock_segment]), mock_info)

            mock_model.transcribe.side_effect = make_transcribe_result
            mock_model_class.return_value = mock_model

            transcribe_directory(
                audio_dir=audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

            # Verify model was created with correct params
            mock_model_class.assert_called_once_with(
                "medium", device="cuda", compute_type="float16"
            )


class TestTranscribeIntegration:
    """Integration tests for transcription workflow."""

    def test_full_transcription_workflow(
        self, temp_audio_dir: Path, temp_output_dir: Path
    ) -> None:
        """Test complete transcription workflow from audio to outputs."""
        # Setup: Create audio files
        video_ids = ["video_001", "video_002", "video_003"]
        for vid in video_ids:
            (temp_audio_dir / f"{vid}.m4a").write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(
            model="tiny",
            device="cpu",
            compute_type="int8",
            skip_existing=True,
        )

        with patch("faster_whisper.WhisperModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.model_path = "tiny"  # Required for JSON serialization
            
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.duration = 5.0

            def make_transcribe_result(*args, **kwargs):
                mock_segment = MagicMock()
                mock_segment.start = 0.0
                mock_segment.end = 5.0
                mock_segment.text = "Transcribed content"
                return (iter([mock_segment]), mock_info)

            mock_model.transcribe.side_effect = make_transcribe_result
            mock_model_class.return_value = mock_model

            results = transcribe_directory(
                audio_dir=temp_audio_dir,
                out_dir=temp_output_dir,
                config=config,
            )

        # Verify results
        assert len(results) == 3

        # Verify output files exist
        for vid in video_ids:
            assert (temp_output_dir / f"{vid}.json").exists()
            assert (temp_output_dir / f"{vid}.txt").exists()

        # Verify JSON structure
        for vid in video_ids:
            data = json.loads((temp_output_dir / f"{vid}.json").read_text())
            assert data["video_id"] == vid
            assert len(data["segments"]) == 1
            assert data["segments"][0]["text"] == "Transcribed content"


# =============================================================================
# SRT/VTT FORMAT TESTS
# =============================================================================


class TestTimestampFormatting:
    """Tests for timestamp formatting functions."""

    def test_format_timestamp_srt_zero(self) -> None:
        """Verify zero seconds formats correctly for SRT."""
        assert _format_timestamp_srt(0.0) == "00:00:00,000"

    def test_format_timestamp_srt_simple(self) -> None:
        """Verify simple timestamps format correctly for SRT."""
        assert _format_timestamp_srt(5.5) == "00:00:05,500"
        assert _format_timestamp_srt(65.123) == "00:01:05,123"

    def test_format_timestamp_srt_hours(self) -> None:
        """Verify timestamps with hours format correctly for SRT."""
        assert _format_timestamp_srt(3661.5) == "01:01:01,500"
        assert _format_timestamp_srt(7200.0) == "02:00:00,000"

    def test_format_timestamp_vtt_zero(self) -> None:
        """Verify zero seconds formats correctly for VTT."""
        assert _format_timestamp_vtt(0.0) == "00:00:00.000"

    def test_format_timestamp_vtt_simple(self) -> None:
        """Verify simple timestamps format correctly for VTT."""
        assert _format_timestamp_vtt(5.5) == "00:00:05.500"
        assert _format_timestamp_vtt(65.123) == "00:01:05.123"

    def test_format_timestamp_vtt_hours(self) -> None:
        """Verify timestamps with hours format correctly for VTT."""
        assert _format_timestamp_vtt(3661.5) == "01:01:01.500"
        assert _format_timestamp_vtt(7200.0) == "02:00:00.000"

    def test_srt_uses_comma_separator(self) -> None:
        """Verify SRT uses comma for milliseconds separator."""
        result = _format_timestamp_srt(1.234)
        assert "," in result
        assert "." not in result

    def test_vtt_uses_dot_separator(self) -> None:
        """Verify VTT uses dot for milliseconds separator."""
        result = _format_timestamp_vtt(1.234)
        assert "." in result
        assert "," not in result


class TestSegmentsToSrt:
    """Tests for segments_to_srt function."""

    def test_empty_segments(self) -> None:
        """Verify empty segments produce empty SRT."""
        result = segments_to_srt([])
        assert result == ""

    def test_single_segment(self) -> None:
        """Verify single segment produces valid SRT."""
        segments = [Segment(start=0.0, end=5.0, text="Hello world")]
        result = segments_to_srt(segments)

        assert "1\n" in result
        assert "00:00:00,000 --> 00:00:05,000" in result
        assert "Hello world" in result

    def test_multiple_segments(self) -> None:
        """Verify multiple segments produce valid SRT with sequential numbers."""
        segments = [
            Segment(start=0.0, end=5.0, text="First subtitle"),
            Segment(start=5.0, end=10.0, text="Second subtitle"),
            Segment(start=10.0, end=15.0, text="Third subtitle"),
        ]
        result = segments_to_srt(segments)

        assert "1\n" in result
        assert "2\n" in result
        assert "3\n" in result
        assert "First subtitle" in result
        assert "Second subtitle" in result
        assert "Third subtitle" in result

    def test_srt_format_structure(self) -> None:
        """Verify SRT output has correct structure."""
        segments = [Segment(start=0.0, end=5.0, text="Test")]
        result = segments_to_srt(segments)
        lines = result.split("\n")

        # SRT format: index, timestamp, text, blank line
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert lines[2] == "Test"
        assert lines[3] == ""

    def test_srt_preserves_text(self) -> None:
        """Verify SRT preserves segment text exactly."""
        text = "Hello, this is a test with punctuation!"
        segments = [Segment(start=0.0, end=5.0, text=text)]
        result = segments_to_srt(segments)

        assert text in result


class TestSegmentsToVtt:
    """Tests for segments_to_vtt function."""

    def test_empty_segments(self) -> None:
        """Verify empty segments produce VTT header only."""
        result = segments_to_vtt([])
        assert result.startswith("WEBVTT")

    def test_vtt_header(self) -> None:
        """Verify VTT output starts with WEBVTT header."""
        segments = [Segment(start=0.0, end=5.0, text="Hello")]
        result = segments_to_vtt(segments)

        assert result.startswith("WEBVTT")

    def test_single_segment(self) -> None:
        """Verify single segment produces valid VTT."""
        segments = [Segment(start=0.0, end=5.0, text="Hello world")]
        result = segments_to_vtt(segments)

        assert "00:00:00.000 --> 00:00:05.000" in result
        assert "Hello world" in result

    def test_multiple_segments(self) -> None:
        """Verify multiple segments produce valid VTT."""
        segments = [
            Segment(start=0.0, end=5.0, text="First subtitle"),
            Segment(start=5.0, end=10.0, text="Second subtitle"),
        ]
        result = segments_to_vtt(segments)

        assert "First subtitle" in result
        assert "Second subtitle" in result

    def test_vtt_no_index_numbers(self) -> None:
        """Verify VTT does not include index numbers (unlike SRT)."""
        segments = [
            Segment(start=0.0, end=5.0, text="First"),
            Segment(start=5.0, end=10.0, text="Second"),
        ]
        result = segments_to_vtt(segments)
        lines = result.split("\n")

        # VTT doesn't require index numbers
        # Check that lines after header are timestamps or text
        for line in lines[2:]:  # Skip WEBVTT and blank line
            if line.strip():
                assert "-->" in line or not line[0].isdigit() or len(line) > 5


class TestOutputFormats:
    """Tests for output format configuration."""

    def test_default_output_formats(self) -> None:
        """Verify default output formats are json and txt."""
        config = TranscriptionConfig()
        assert "json" in config.output_formats
        assert "txt" in config.output_formats

    def test_custom_output_formats(self) -> None:
        """Verify custom output formats can be set."""
        config = TranscriptionConfig(output_formats={"srt", "vtt"})
        assert "srt" in config.output_formats
        assert "vtt" in config.output_formats
        assert "json" not in config.output_formats
        assert "txt" not in config.output_formats

    def test_all_formats(self) -> None:
        """Verify all formats can be enabled together."""
        config = TranscriptionConfig(output_formats={"json", "txt", "srt", "vtt"})
        assert len(config.output_formats) == 4


class TestTranscribeFileWithFormats:
    """Tests for transcribe_file with different output formats."""

    def test_srt_output_created(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
    ) -> None:
        """Verify SRT file is created when format is requested."""
        audio_file = temp_audio_dir / "test_srt.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(
            output_formats={"srt"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.srt_path is not None
        assert result.srt_path.exists()
        assert result.json_path is None
        assert result.txt_path is None

        # Verify SRT content
        content = result.srt_path.read_text()
        assert "1\n" in content
        assert "-->" in content

    def test_vtt_output_created(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
    ) -> None:
        """Verify VTT file is created when format is requested."""
        audio_file = temp_audio_dir / "test_vtt.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(
            output_formats={"vtt"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.vtt_path is not None
        assert result.vtt_path.exists()

        # Verify VTT content
        content = result.vtt_path.read_text()
        assert content.startswith("WEBVTT")

    def test_all_formats_created(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
    ) -> None:
        """Verify all format files are created when all formats requested."""
        audio_file = temp_audio_dir / "test_all.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        config = TranscriptionConfig(
            output_formats={"json", "txt", "srt", "vtt"},
            skip_existing=False,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        assert result is not None
        assert result.json_path is not None and result.json_path.exists()
        assert result.txt_path is not None and result.txt_path.exists()
        assert result.srt_path is not None and result.srt_path.exists()
        assert result.vtt_path is not None and result.vtt_path.exists()

    def test_skip_existing_with_srt_vtt(
        self,
        mock_whisper_model: MagicMock,
        temp_audio_dir: Path,
        temp_output_dir: Path,
    ) -> None:
        """Verify skip_existing works with SRT/VTT formats."""
        stem = "test_skip_srt"
        audio_file = temp_audio_dir / f"{stem}.m4a"
        audio_file.write_bytes(b"\x00" * 1024)

        # Create existing SRT file
        srt_path = temp_output_dir / f"{stem}.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nExisting\n\n")

        config = TranscriptionConfig(
            output_formats={"srt"},
            skip_existing=True,
        )

        result = transcribe_file(
            model=mock_whisper_model,
            audio_path=audio_file,
            out_dir=temp_output_dir,
            config=config,
            device="cpu",
            compute_type="int8",
        )

        # Should be skipped
        assert result is None
        mock_whisper_model.transcribe.assert_not_called()

