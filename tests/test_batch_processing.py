"""
Comprehensive tests for batch processing and parallelization strategies.

This module tests the Strategy Pattern implementation for parallel transcription,
including GPU detection, worker allocation, and different processing strategies.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from ytpipe.batch_processing import (
    GPUInfo,
    MultiGPUStrategy,
    ParallelCPUStrategy,
    SequentialStrategy,
    TranscriptionStrategyFactory,
    detect_available_gpus,
    get_optimal_worker_count,
)
from ytpipe.config import TranscriptionConfig
from ytpipe.transcribe import Segment, TranscriptionResult


# =============================================================================
# GPU DETECTION TESTS
# =============================================================================


class TestDetectAvailableGPUs:
    """Tests for GPU detection using nvidia-smi."""

    def test_detects_single_gpu(self):
        """Should parse nvidia-smi output for single GPU."""
        mock_output = "0, NVIDIA GeForce RTX 3090, 24576\n"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout=mock_output, returncode=0
            )

            gpus = detect_available_gpus()

            assert len(gpus) == 1
            assert gpus[0].device_id == 0
            assert "RTX 3090" in gpus[0].name
            assert gpus[0].memory_total_mb == 24576

    def test_detects_multiple_gpus(self):
        """Should detect and parse multiple GPUs."""
        mock_output = (
            "0, NVIDIA GeForce RTX 3090, 24576\n"
            "1, NVIDIA GeForce RTX 3080, 10240\n"
            "2, NVIDIA Tesla V100, 32768\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout=mock_output, returncode=0
            )

            gpus = detect_available_gpus()

            assert len(gpus) == 3
            assert gpus[0].device_id == 0
            assert gpus[1].device_id == 1
            assert gpus[2].device_id == 2
            assert "V100" in gpus[2].name

    def test_returns_empty_when_nvidia_smi_not_found(self):
        """Should return empty list when nvidia-smi is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_available_gpus()
            assert gpus == []

    def test_returns_empty_when_nvidia_smi_fails(self):
        """Should return empty list when nvidia-smi command fails."""
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            gpus = detect_available_gpus()
            assert gpus == []

    def test_returns_empty_when_nvidia_smi_times_out(self):
        """Should return empty list when nvidia-smi times out."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvidia-smi", 5)):
            gpus = detect_available_gpus()
            assert gpus == []

    def test_handles_empty_output(self):
        """Should handle empty nvidia-smi output gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)

            gpus = detect_available_gpus()
            assert gpus == []

    def test_handles_malformed_output(self):
        """Should skip malformed lines in nvidia-smi output."""
        mock_output = (
            "0, NVIDIA GeForce RTX 3090, 24576\n"
            "invalid line without commas\n"
            "1, NVIDIA GeForce RTX 3080, 10240\n"
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout=mock_output, returncode=0
            )

            gpus = detect_available_gpus()

            # Should successfully parse 2 valid lines and skip invalid one
            assert len(gpus) == 2


class TestGetOptimalWorkerCount:
    """Tests for optimal worker count determination."""

    def test_respects_user_specified_count(self):
        """Should use user-specified worker count if provided."""
        assert get_optimal_worker_count("cpu", num_workers=8) == 8
        assert get_optimal_worker_count("cuda", num_workers=4) == 4

    def test_cpu_defaults_to_cores_minus_one(self):
        """Should default to CPU count - 1 for CPU device."""
        with patch("multiprocessing.cpu_count", return_value=8):
            assert get_optimal_worker_count("cpu") == 7

    def test_cpu_minimum_one_worker(self):
        """Should return at least 1 worker even on single-core systems."""
        with patch("multiprocessing.cpu_count", return_value=1):
            assert get_optimal_worker_count("cpu") == 1

    def test_cuda_defaults_to_one(self):
        """Should default to 1 worker for CUDA (GPU-bound workload)."""
        assert get_optimal_worker_count("cuda") == 1

    def test_handles_none_cpu_count(self):
        """Should handle case where cpu_count() returns None."""
        with patch("multiprocessing.cpu_count", return_value=None):
            assert get_optimal_worker_count("cpu") == 1


# =============================================================================
# STRATEGY TESTS
# =============================================================================


class TestSequentialStrategy:
    """Tests for SequentialStrategy (single-threaded processing)."""

    def test_strategy_name(self):
        """Should return correct strategy name."""
        config = TranscriptionConfig()
        strategy = SequentialStrategy(config)
        assert strategy.get_strategy_name() == "Sequential (1 file at a time)"

    def test_processes_files_sequentially(self, tmp_path: Path):
        """Should process files one at a time with single model instance."""
        config = TranscriptionConfig(model="tiny")
        strategy = SequentialStrategy(config)

        # Create mock audio files
        audio_files = [
            tmp_path / "audio1.m4a",
            tmp_path / "audio2.m4a",
            tmp_path / "audio3.m4a",
        ]
        for f in audio_files:
            f.write_text("fake audio")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        # Mock transcribe function
        mock_results = [
            TranscriptionResult(
                audio_path=f,
                json_path=out_dir / f"{f.stem}.json",
                txt_path=out_dir / f"{f.stem}.txt",
                srt_path=out_dir / f"{f.stem}.srt",
                vtt_path=out_dir / f"{f.stem}.vtt",
                language="en",
                duration=10.0,
                segments=[Segment(0.0, 5.0, "Test")],
            )
            for f in audio_files
        ]

        call_count = 0

        def mock_transcribe(model, audio_path, out_dir, config, device, compute_type):
            nonlocal call_count
            result = mock_results[call_count]
            call_count += 1
            return result

        # Mock WhisperModel
        with patch("faster_whisper.WhisperModel") as MockModel:
            mock_model = MagicMock()
            MockModel.return_value = mock_model

            results = strategy.process_files(
                audio_files=audio_files,
                out_dir=out_dir,
                transcribe_fn=mock_transcribe,
                device="cpu",
                compute_type="int8",
            )

            # Should create only one model instance
            assert MockModel.call_count == 1

            # Should process all files
            assert len(results) == 3
            assert call_count == 3

    def test_continues_on_error(self, tmp_path: Path):
        """Should continue processing remaining files if one fails."""
        config = TranscriptionConfig()
        strategy = SequentialStrategy(config)

        audio_files = [
            tmp_path / "audio1.m4a",
            tmp_path / "audio2.m4a",
            tmp_path / "audio3.m4a",
        ]
        for f in audio_files:
            f.write_text("fake")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        call_count = 0

        def mock_transcribe(model, audio_path, out_dir, config, device, compute_type):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated error")
            return TranscriptionResult(
                audio_path=audio_path,
                json_path=None,
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                language=None,
                duration=None,
                segments=[],
            )

        with patch("faster_whisper.WhisperModel"):
            results = strategy.process_files(
                audio_files=audio_files,
                out_dir=out_dir,
                transcribe_fn=mock_transcribe,
                device="cpu",
                compute_type="int8",
            )

            # Should process 3 files but only return 2 results (1 failed)
            assert call_count == 3
            assert len(results) == 2


class TestParallelCPUStrategy:
    """Tests for ParallelCPUStrategy (multi-process CPU parallelization)."""

    def test_strategy_name(self):
        """Should return strategy name with worker count."""
        config = TranscriptionConfig(num_workers=4)
        strategy = ParallelCPUStrategy(config)

        with patch("ytpipe.batch_processing.get_optimal_worker_count", return_value=4):
            name = strategy.get_strategy_name()
            assert "4 workers" in name

    def test_uses_multiple_workers(self):
        """Should use ProcessPoolExecutor with correct worker count."""
        config = TranscriptionConfig(num_workers=4)
        strategy = ParallelCPUStrategy(config)

        # Test that strategy configures correct number of workers
        with patch("ytpipe.batch_processing.get_optimal_worker_count", return_value=4) as mock_worker_count:
            with patch("ytpipe.batch_processing.ProcessPoolExecutor") as MockExecutor:
                # Mock the executor context manager
                mock_executor_instance = MagicMock()
                MockExecutor.return_value.__enter__.return_value = mock_executor_instance
                mock_executor_instance.submit.return_value = MagicMock()

                # Mock as_completed to return immediately
                with patch("ytpipe.batch_processing.as_completed", return_value=[]):
                    strategy.process_files(
                        audio_files=[],
                        out_dir=Path("/fake"),
                        transcribe_fn=lambda *args, **kwargs: None,
                        device="cpu",
                        compute_type="int8",
                    )

                # Verify ProcessPoolExecutor was created with correct worker count
                MockExecutor.assert_called_once_with(max_workers=4)


class TestMultiGPUStrategy:
    """Tests for MultiGPUStrategy (multi-GPU parallelization)."""

    def test_strategy_name(self):
        """Should return strategy name with GPU count."""
        config = TranscriptionConfig()
        strategy = MultiGPUStrategy(config)

        mock_gpus = [
            GPUInfo(0, "GPU 0", 8192),
            GPUInfo(1, "GPU 1", 8192),
        ]

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=mock_gpus):
            name = strategy.get_strategy_name()
            assert "2 GPUs" in name

    def test_falls_back_to_sequential_when_no_gpus(self, tmp_path: Path):
        """Should fall back to sequential strategy if no GPUs detected."""
        config = TranscriptionConfig()
        strategy = MultiGPUStrategy(config)

        audio_files = [tmp_path / "audio1.m4a"]
        audio_files[0].write_text("fake")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        def mock_transcribe(model, audio_path, out_dir, config, device, compute_type):
            return TranscriptionResult(
                audio_path=audio_path,
                json_path=None,
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                language=None,
                duration=None,
                segments=[],
            )

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=[]):
            with patch("faster_whisper.WhisperModel"):
                results = strategy.process_files(
                    audio_files=audio_files,
                    out_dir=out_dir,
                    transcribe_fn=mock_transcribe,
                    device="cpu",
                    compute_type="int8",
                )

                assert len(results) == 1

    def test_distributes_files_across_gpus(self, tmp_path: Path):
        """Should distribute files evenly across available GPUs (round-robin)."""
        config = TranscriptionConfig()
        strategy = MultiGPUStrategy(config)

        # 5 files, 2 GPUs -> GPU0: 3 files, GPU1: 2 files
        audio_files = [tmp_path / f"audio{i}.m4a" for i in range(5)]
        for f in audio_files:
            f.write_text("fake")

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        mock_gpus = [
            GPUInfo(0, "GPU 0", 8192),
            GPUInfo(1, "GPU 1", 8192),
        ]

        def mock_transcribe(model, audio_path, out_dir, config, device, compute_type):
            return TranscriptionResult(
                audio_path=audio_path,
                json_path=None,
                txt_path=None,
                srt_path=None,
                vtt_path=None,
                language=None,
                duration=None,
                segments=[],
            )

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=mock_gpus):
            with patch("faster_whisper.WhisperModel"):
                with patch.dict("os.environ", {}, clear=False):
                    results = strategy.process_files(
                        audio_files=audio_files,
                        out_dir=out_dir,
                        transcribe_fn=mock_transcribe,
                        device="cuda",
                        compute_type="float16",
                    )

                    # Should process all 5 files
                    assert len(results) == 5


# =============================================================================
# STRATEGY FACTORY TESTS
# =============================================================================


class TestTranscriptionStrategyFactory:
    """Tests for TranscriptionStrategyFactory (Strategy selection)."""

    def test_creates_sequential_strategy_by_default(self):
        """Should create SequentialStrategy for single worker."""
        config = TranscriptionConfig(num_workers=1)
        strategy = TranscriptionStrategyFactory.create_strategy(config, "cpu")

        assert isinstance(strategy, SequentialStrategy)

    def test_creates_parallel_cpu_strategy(self):
        """Should create ParallelCPUStrategy for multi-worker CPU."""
        config = TranscriptionConfig(num_workers=4)
        strategy = TranscriptionStrategyFactory.create_strategy(config, "cpu")

        assert isinstance(strategy, ParallelCPUStrategy)

    def test_creates_multi_gpu_strategy_when_multiple_gpus_available(self):
        """Should create MultiGPUStrategy when multiple GPUs detected."""
        config = TranscriptionConfig(num_workers=2)

        mock_gpus = [
            GPUInfo(0, "GPU 0", 8192),
            GPUInfo(1, "GPU 1", 8192),
        ]

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=mock_gpus):
            strategy = TranscriptionStrategyFactory.create_strategy(config, "cuda")

            assert isinstance(strategy, MultiGPUStrategy)

    def test_creates_sequential_strategy_for_single_gpu(self):
        """Should create SequentialStrategy for single GPU even with multiple workers."""
        config = TranscriptionConfig(num_workers=2)

        mock_gpus = [GPUInfo(0, "GPU 0", 8192)]

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=mock_gpus):
            strategy = TranscriptionStrategyFactory.create_strategy(config, "cuda")

            assert isinstance(strategy, SequentialStrategy)

    def test_creates_sequential_strategy_for_cuda_single_worker(self):
        """Should create SequentialStrategy for CUDA with single worker."""
        config = TranscriptionConfig(num_workers=1)

        with patch("ytpipe.batch_processing.detect_available_gpus", return_value=[]):
            strategy = TranscriptionStrategyFactory.create_strategy(config, "cuda")

            assert isinstance(strategy, SequentialStrategy)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestBatchProcessingIntegration:
    """Integration tests for batch processing workflow."""

    def test_config_with_parallelization_options(self):
        """Should create config with parallelization options."""
        config = TranscriptionConfig(
            model="medium",
            device="cpu",
            num_workers=4,
            batch_size=2,
            max_queue_size=10,
        )

        assert config.num_workers == 4
        assert config.batch_size == 2
        assert config.max_queue_size == 10

    def test_default_parallelization_config(self):
        """Should have sensible defaults for parallelization."""
        config = TranscriptionConfig()

        assert config.num_workers == 1  # Sequential by default
        assert config.batch_size == 1
        assert config.max_queue_size == 10
