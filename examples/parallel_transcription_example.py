"""
Example: Using Parallel Transcription

This example demonstrates how to use ytpipe's batch processing capabilities
for faster transcription using parallel CPU workers or multiple GPUs.
"""

from pathlib import Path

from ytpipe.config import TranscriptionConfig
from ytpipe.transcribe import transcribe_directory


def example_sequential():
    """Example 1: Sequential processing (default, safest)."""
    print("=" * 60)
    print("Example 1: Sequential Processing")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        device="auto",
        num_workers=1,  # Sequential processing
        output_formats={"json", "txt", "srt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files sequentially")


def example_parallel_cpu():
    """Example 2: Parallel CPU processing for faster transcription."""
    print("\n" + "=" * 60)
    print("Example 2: Parallel CPU Processing (4 workers)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="small",  # Use smaller model for faster CPU processing
        device="cpu",
        compute_type="int8",  # Optimized for CPU
        num_workers=4,  # 4 parallel workers
        output_formats={"json", "txt", "srt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files using 4 CPU workers")
    print("Expected speedup: 3-4x faster than sequential")


def example_multi_gpu():
    """Example 3: Multi-GPU processing for maximum throughput."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-GPU Processing")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",  # Large model benefits from GPU
        device="cuda",
        compute_type="float16",  # Optimized for GPU
        num_workers=2,  # Use 2 GPUs
        vad_filter=True,  # Enable VAD for better quality
        output_formats={"json", "txt", "srt", "vtt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files using multi-GPU")
    print("GPUs detected automatically and work distributed evenly")


def example_auto_optimization():
    """Example 4: Auto-optimization - let ytpipe choose the best strategy."""
    print("\n" + "=" * 60)
    print("Example 4: Auto-Optimization")
    print("=" * 60)

    # Auto-detect device (CUDA if available, else CPU)
    # Auto-detect optimal worker count
    config = TranscriptionConfig(
        model="medium",
        device="auto",  # Auto-detect best device
        compute_type="auto",  # Auto-select compute type
        # num_workers not specified = auto-detect optimal count
        output_formats={"json", "srt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with auto-optimization")
    print("Strategy selected automatically based on hardware")


def example_custom_configuration():
    """Example 5: Custom configuration for specific use case."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        beam_size=5,  # Higher beam size for better quality
        vad_filter=True,  # Filter out silence
        language="en",  # English hint for better accuracy
        skip_existing=True,  # Skip already transcribed files
        num_workers=2,
        output_formats={"json", "srt", "vtt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with custom config")
    print("High quality settings with parallel processing")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ytpipe Parallel Transcription Examples")
    print("=" * 60)

    # Run examples based on what makes sense for your system
    # Comment out examples you don't want to run

    # Example 1: Sequential (always works)
    example_sequential()

    # Example 2: Parallel CPU (good for multi-core systems)
    # example_parallel_cpu()

    # Example 3: Multi-GPU (requires 2+ GPUs)
    # example_multi_gpu()

    # Example 4: Auto-optimization (recommended)
    # example_auto_optimization()

    # Example 5: Custom configuration
    # example_custom_configuration()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
