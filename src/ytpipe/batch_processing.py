"""
Batch processing and parallelization strategies for transcription.

This module provides a clean, extensible architecture for processing audio files
in parallel using different strategies (sequential, multi-process, multi-GPU).

Design Patterns:
    - Strategy Pattern: Different processing strategies (Sequential, Parallel, MultiGPU)
    - Factory Pattern: Strategy selection based on configuration
    - Template Method: Common processing workflow with strategy-specific execution
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from tqdm import tqdm

from .config import TranscriptionConfig

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from .transcribe import TranscriptionResult

logger = logging.getLogger(__name__)


# =============================================================================
# GPU DETECTION AND UTILITIES
# =============================================================================


@dataclass(frozen=True, slots=True)
class GPUInfo:
    """Information about available GPU device."""

    device_id: int
    name: str
    memory_total_mb: int


def detect_available_gpus() -> list[GPUInfo]:
    """
    Detect all available NVIDIA GPUs using nvidia-smi.

    Returns
    -------
    list[GPUInfo]
        List of detected GPUs. Empty list if no GPUs or nvidia-smi unavailable.

    Examples
    --------
    >>> gpus = detect_available_gpus()
    >>> if gpus:
    ...     print(f"Found {len(gpus)} GPU(s)")
    ... else:
    ...     print("No GPUs detected")
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        gpus: list[GPUInfo] = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    GPUInfo(
                        device_id=int(parts[0]),
                        name=parts[1],
                        memory_total_mb=int(parts[2]),
                    )
                )

        logger.info("Detected %d GPU(s): %s", len(gpus), [g.name for g in gpus])
        return gpus

    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.debug("GPU detection failed (nvidia-smi not available or errored)")
        return []


def get_optimal_worker_count(device: str, num_workers: int | None = None) -> int:
    """
    Determine optimal number of workers based on device and CPU cores.

    Parameters
    ----------
    device : str
        Device type ("cuda", "cpu", "auto").
    num_workers : int | None
        User-specified worker count. If None, auto-detect.

    Returns
    -------
    int
        Optimal number of workers (minimum 1).

    Notes
    -----
    - For CPU: defaults to CPU count - 1 (leave one core free)
    - For CUDA: defaults to 1 (GPU work is often memory-bound)
    - For multi-GPU: returns number of available GPUs

    Examples
    --------
    >>> get_optimal_worker_count("cpu")
    7  # On 8-core machine
    >>> get_optimal_worker_count("cuda")
    1
    """
    if num_workers is not None and num_workers > 0:
        return num_workers

    if device == "cpu":
        # Use CPU count - 1 to leave one core free
        cpu_count = mp.cpu_count() or 1
        return max(1, cpu_count - 1)

    # For CUDA, default to 1 worker (GPU-bound)
    # Multi-GPU case is handled separately
    return 1


# =============================================================================
# BATCH PROCESSING STRATEGIES (Strategy Pattern)
# =============================================================================


class TranscriptionStrategy(ABC):
    """
    Abstract base class for transcription processing strategies.

    This defines the interface that all processing strategies must implement.
    Follows the Strategy Pattern for flexible algorithm selection.
    """

    def __init__(self, config: TranscriptionConfig):
        """
        Initialize strategy with configuration.

        Parameters
        ----------
        config : TranscriptionConfig
            Transcription configuration settings.
        """
        self.config = config

    @abstractmethod
    def process_files(
        self,
        audio_files: list[Path],
        out_dir: Path,
        transcribe_fn: Callable,
        device: str,
        compute_type: str,
    ) -> list[TranscriptionResult]:
        """
        Process audio files according to strategy implementation.

        Parameters
        ----------
        audio_files : list[Path]
            List of audio file paths to process.
        out_dir : Path
            Output directory for transcriptions.
        transcribe_fn : Callable
            Function to transcribe a single file.
        device : str
            Resolved device ("cuda" or "cpu").
        compute_type : str
            Resolved compute type (e.g., "float16", "int8").

        Returns
        -------
        list[TranscriptionResult]
            List of transcription results.
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """
        Get human-readable name of this strategy.

        Returns
        -------
        str
            Strategy name for logging purposes.
        """
        pass


class SequentialStrategy(TranscriptionStrategy):
    """
    Sequential processing strategy - processes files one at a time.

    This is the simplest and most memory-efficient strategy, suitable for:
    - Single GPU with limited memory
    - Small number of files
    - Debugging and testing
    """

    def get_strategy_name(self) -> str:
        return "Sequential (1 file at a time)"

    def process_files(
        self,
        audio_files: list[Path],
        out_dir: Path,
        transcribe_fn: Callable,
        device: str,
        compute_type: str,
    ) -> list[TranscriptionResult]:
        """Process files sequentially with single model instance."""
        from faster_whisper import WhisperModel

        logger.info(
            "Loading model: %s (device=%s, compute=%s)",
            self.config.model,
            device,
            compute_type,
        )

        model = WhisperModel(
            self.config.model, device=device, compute_type=compute_type
        )

        results: list[TranscriptionResult] = []

        for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
            try:
                result = transcribe_fn(
                    model=model,
                    audio_path=audio_path,
                    out_dir=out_dir,
                    config=self.config,
                    device=device,
                    compute_type=compute_type,
                )
                if result is not None:
                    results.append(result)

            except Exception:
                logger.exception("Failed to transcribe: %s", audio_path)

        return results


# Module-level worker function for CPU parallelization (must be picklable)
def _cpu_worker(
    audio_path: Path,
    out_dir: Path,
    model_name: str,
    compute_type: str,
    config_dict: dict,
) -> TranscriptionResult | None:
    """
    Worker function for CPU parallel processing.

    Must be module-level for pickle serialization (required by multiprocessing).

    Parameters
    ----------
    audio_path : Path
        Path to audio file to transcribe.
    out_dir : Path
        Output directory for transcription results.
    model_name : str
        Whisper model name.
    compute_type : str
        Compute type for model.
    config_dict : dict
        Serialized TranscriptionConfig as dict.

    Returns
    -------
    TranscriptionResult | None
        Transcription result or None if failed.
    """
    try:
        from faster_whisper import WhisperModel

        from .config import TranscriptionConfig
        from .transcribe import transcribe_file

        # Reconstruct config from dict
        config = TranscriptionConfig(**config_dict)

        # Load model in this worker process
        model = WhisperModel(model_name, device="cpu", compute_type=compute_type)

        # Transcribe
        result = transcribe_file(
            model=model,
            audio_path=audio_path,
            out_dir=out_dir,
            config=config,
            device="cpu",
            compute_type=compute_type,
        )
        return result

    except Exception as e:
        logger.exception("Worker failed for file: %s", audio_path)
        return None


class ParallelCPUStrategy(TranscriptionStrategy):
    """
    Multi-process CPU parallelization strategy.

    Uses ProcessPoolExecutor to process multiple files simultaneously on CPU.
    Each worker loads its own model instance in a separate process.

    Suitable for:
    - CPU-only environments
    - Large batch of files
    - Multi-core machines
    """

    def get_strategy_name(self) -> str:
        num_workers = get_optimal_worker_count("cpu", self.config.num_workers)
        return f"Parallel CPU ({num_workers} workers)"

    def process_files(
        self,
        audio_files: list[Path],
        out_dir: Path,
        transcribe_fn: Callable,
        device: str,
        compute_type: str,
    ) -> list[TranscriptionResult]:
        """Process files in parallel using multiple CPU processes."""
        num_workers = get_optimal_worker_count("cpu", self.config.num_workers)

        logger.info(
            "Using %d worker processes for CPU parallelization", num_workers
        )

        # Serialize config for pickling
        config_dict = {
            "model": self.config.model,
            "device": self.config.device,
            "compute_type": self.config.compute_type,
            "beam_size": self.config.beam_size,
            "vad_filter": self.config.vad_filter,
            "vad_parameters": self.config.vad_parameters,
            "language": self.config.language,
            "audio_extensions": self.config.audio_extensions,
            "skip_existing": self.config.skip_existing,
            "output_formats": self.config.output_formats,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "max_queue_size": self.config.max_queue_size,
        }

        results: list[TranscriptionResult] = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _cpu_worker,
                    audio_file,
                    out_dir,
                    self.config.model,
                    compute_type,
                    config_dict,
                ): audio_file
                for audio_file in audio_files
            }

            # Collect results with progress bar
            with tqdm(total=len(audio_files), desc="Transcribing", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        audio_file = futures[future]
                        logger.exception("Failed to process: %s", audio_file)
                    finally:
                        pbar.update(1)

        return results


class MultiGPUStrategy(TranscriptionStrategy):
    """
    Multi-GPU parallelization strategy.

    Distributes work across multiple NVIDIA GPUs. Each GPU gets its own
    worker thread with a dedicated model instance.

    Suitable for:
    - Multi-GPU machines (e.g., 2+ GPUs)
    - Large batches of files
    - Maximum throughput requirements

    Architecture:
    - Each GPU gets 1 worker thread (GPU work is memory-bound, not CPU-bound)
    - Files are distributed round-robin across GPUs
    - Uses ThreadPoolExecutor (not ProcessPoolExecutor) for lower overhead
    """

    def get_strategy_name(self) -> str:
        gpus = detect_available_gpus()
        return f"Multi-GPU ({len(gpus)} GPUs)"

    def process_files(
        self,
        audio_files: list[Path],
        out_dir: Path,
        transcribe_fn: Callable,
        device: str,
        compute_type: str,
    ) -> list[TranscriptionResult]:
        """Process files across multiple GPUs in parallel."""
        gpus = detect_available_gpus()

        if not gpus:
            logger.warning("No GPUs detected, falling back to sequential processing")
            return SequentialStrategy(self.config).process_files(
                audio_files, out_dir, transcribe_fn, "cpu", "int8"
            )

        logger.info("Distributing work across %d GPU(s)", len(gpus))

        # Pre-load models for each GPU (thread-safe, done in main thread)
        models_per_gpu: dict[int, WhisperModel] = {}

        for gpu in gpus:
            logger.info(
                "Loading model on GPU %d (%s)", gpu.device_id, gpu.name
            )
            # Use CUDA_VISIBLE_DEVICES to isolate each model to specific GPU
            original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu.device_id)

                from faster_whisper import WhisperModel

                models_per_gpu[gpu.device_id] = WhisperModel(
                    self.config.model, device="cuda", compute_type=compute_type
                )
            finally:
                if original_visible is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
                else:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Distribute files across GPUs (round-robin)
        files_per_gpu: dict[int, list[Path]] = {gpu.device_id: [] for gpu in gpus}

        for i, audio_file in enumerate(audio_files):
            gpu_id = gpus[i % len(gpus)].device_id
            files_per_gpu[gpu_id].append(audio_file)

        logger.info(
            "File distribution: %s",
            {f"GPU{k}": len(v) for k, v in files_per_gpu.items()},
        )

        def worker_fn(gpu_id: int, audio_path: Path) -> TranscriptionResult | None:
            """Worker function for specific GPU."""
            try:
                # Set GPU for this thread
                original_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                try:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

                    model = models_per_gpu[gpu_id]
                    result = transcribe_fn(
                        model=model,
                        audio_path=audio_path,
                        out_dir=out_dir,
                        config=self.config,
                        device="cuda",
                        compute_type=compute_type,
                    )
                    return result

                finally:
                    if original_visible is not None:
                        os.environ["CUDA_VISIBLE_DEVICES"] = original_visible
                    else:
                        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            except Exception:
                logger.exception(
                    "GPU %d worker failed for file: %s", gpu_id, audio_path
                )
                return None

        results: list[TranscriptionResult] = []

        # Use ThreadPoolExecutor (not ProcessPoolExecutor)
        # GPU operations release GIL, so threads are efficient
        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            # Submit all tasks
            futures = []
            for gpu_id, gpu_files in files_per_gpu.items():
                for audio_file in gpu_files:
                    future = executor.submit(worker_fn, gpu_id, audio_file)
                    futures.append(future)

            # Collect results with progress bar
            with tqdm(total=len(audio_files), desc="Transcribing", unit="file") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception:
                        logger.exception("Task failed")
                    finally:
                        pbar.update(1)

        return results


# =============================================================================
# STRATEGY FACTORY (Factory Pattern)
# =============================================================================


class TranscriptionStrategyFactory:
    """
    Factory for creating appropriate transcription strategy based on config.

    Implements the Factory Pattern to encapsulate strategy selection logic.
    """

    @staticmethod
    def create_strategy(
        config: TranscriptionConfig, device: str
    ) -> TranscriptionStrategy:
        """
        Create and return appropriate transcription strategy.

        Selection logic:
        1. If multiple GPUs detected and num_workers > 1 → MultiGPUStrategy
        2. If CPU device and num_workers > 1 → ParallelCPUStrategy
        3. Otherwise → SequentialStrategy

        Parameters
        ----------
        config : TranscriptionConfig
            Transcription configuration.
        device : str
            Resolved device ("cuda" or "cpu").

        Returns
        -------
        TranscriptionStrategy
            Appropriate strategy instance for the configuration.

        Examples
        --------
        >>> config = TranscriptionConfig(num_workers=4, device="cpu")
        >>> strategy = TranscriptionStrategyFactory.create_strategy(config, "cpu")
        >>> isinstance(strategy, ParallelCPUStrategy)
        True
        """
        # Multi-GPU strategy: if CUDA and multiple GPUs available
        if device == "cuda" and config.num_workers > 1:
            gpus = detect_available_gpus()
            if len(gpus) > 1:
                logger.info(
                    "Selected Multi-GPU strategy (%d GPUs detected)", len(gpus)
                )
                return MultiGPUStrategy(config)

        # Parallel CPU strategy: if CPU and multiple workers requested
        if device == "cpu" and config.num_workers > 1:
            logger.info("Selected Parallel CPU strategy")
            return ParallelCPUStrategy(config)

        # Default: Sequential strategy
        logger.info("Selected Sequential strategy")
        return SequentialStrategy(config)
