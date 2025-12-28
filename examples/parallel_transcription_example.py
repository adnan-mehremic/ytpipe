from pathlib import Path

from ytpipe.config import TranscriptionConfig
from ytpipe.transcribe import transcribe_directory


def example_sequential():
    print("=" * 60)
    print("Example 1: Sequential Processing")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        device="auto",
        num_workers=1,
        output_formats={"json", "txt", "srt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files sequentially")


def example_parallel_cpu():
    print("\n" + "=" * 60)
    print("Example 2: Parallel CPU Processing (4 workers)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="small",
        device="cpu",
        compute_type="int8",
        num_workers=4,
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
    print("\n" + "=" * 60)
    print("Example 3: Multi-GPU Processing")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        num_workers=2,
        vad_filter=True,
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
    print("\n" + "=" * 60)
    print("Example 4: Auto-Optimization")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        device="auto",
        compute_type="auto",
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
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        beam_size=5,
        vad_filter=True,
        language="en",
        skip_existing=True,
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
    print("\n" + "=" * 60)
    print("ytpipe Parallel Transcription Examples")
    print("=" * 60)

    example_sequential()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
