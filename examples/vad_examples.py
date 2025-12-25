"""
Example: Voice Activity Detection (VAD) Configuration

This example demonstrates different VAD configurations for various use cases.
VAD significantly improves transcription quality and speed.
"""

from pathlib import Path

from ytpipe.config import TranscriptionConfig, get_default_vad_parameters, validate_vad_parameters
from ytpipe.transcribe import transcribe_directory


def example_default_vad():
    """Example 1: Default VAD (recommended starting point)."""
    print("=" * 60)
    print("Example 1: Default VAD Configuration")
    print("=" * 60)

    # VAD is enabled by default with optimal parameters
    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,  # Default: True
        # vad_parameters not specified = use optimal defaults
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with default VAD")
    print("Benefits: ~20-30% faster, fewer hallucinations, better timestamps")


def example_inspect_defaults():
    """Example 2: Inspect default VAD parameters."""
    print("\n" + "=" * 60)
    print("Example 2: Inspect Default Parameters")
    print("=" * 60)

    params = get_default_vad_parameters()

    print("\nDefault VAD parameters:")
    print(f"  threshold:                {params['threshold']}")
    print(f"  min_speech_duration_ms:   {params['min_speech_duration_ms']}")
    print(f"  max_speech_duration_s:    {params['max_speech_duration_s']}")
    print(f"  min_silence_duration_ms:  {params['min_silence_duration_ms']}")
    print(f"  window_size_samples:      {params['window_size_samples']}")
    print(f"  speech_pad_ms:            {params['speech_pad_ms']}")


def example_aggressive_vad():
    """Example 3: Aggressive VAD for noisy environments."""
    print("\n" + "=" * 60)
    print("Example 3: Aggressive VAD (Noisy Audio)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.7,                # Very strict (only clear speech)
            "min_speech_duration_ms": 500,   # Longer minimum (filter short noises)
            "min_silence_duration_ms": 1000, # Shorter silence (aggressive splitting)
            "speech_pad_ms": 300,
        },
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with aggressive VAD")
    print("Use case: Street interviews, outdoor recordings, background music")


def example_sensitive_vad():
    """Example 4: Sensitive VAD for quiet speech."""
    print("\n" + "=" * 60)
    print("Example 4: Sensitive VAD (Quiet Speech)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.3,                # Very sensitive (catch quiet speech)
            "min_speech_duration_ms": 100,   # Catch short utterances
            "min_silence_duration_ms": 3000, # Allow longer pauses
            "speech_pad_ms": 500,            # Extra padding
        },
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with sensitive VAD")
    print("Use case: ASMR, soft-spoken speakers, low-volume recordings")


def example_podcast_optimized():
    """Example 5: Podcast-optimized VAD."""
    print("\n" + "=" * 60)
    print("Example 5: Podcast-Optimized VAD")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,                 # Balanced
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 120.0,   # Allow 2-minute monologues
            "min_silence_duration_ms": 2000,  # Natural pauses
            "speech_pad_ms": 500,             # Don't cut off words
        },
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with podcast-optimized VAD")
    print("Use case: Podcasts, interviews, long-form conversational content")


def example_disable_vad():
    """Example 6: Disable VAD (transcribe everything including silence)."""
    print("\n" + "=" * 60)
    print("Example 6: Disable VAD")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=False,  # Disable VAD
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files WITHOUT VAD")
    print("Warning: Slower, more hallucinations, less accurate timestamps")
    print("Only use when you need to transcribe all audio including silence")


def example_validate_custom_params():
    """Example 7: Validate custom VAD parameters."""
    print("\n" + "=" * 60)
    print("Example 7: Validate Custom Parameters")
    print("=" * 60)

    # Partial parameters (rest filled with defaults)
    custom_params = {
        "threshold": 0.6,
        "min_speech_duration_ms": 300,
    }

    # Validate and fill in defaults
    validated = validate_vad_parameters(custom_params)

    print("\nCustom parameters:")
    for key, value in custom_params.items():
        print(f"  {key}: {value}")

    print("\nValidated (complete) parameters:")
    for key, value in validated.items():
        print(f"  {key}: {value}")


def example_fast_transcription():
    """Example 8: Fast transcription (speed over accuracy)."""
    print("\n" + "=" * 60)
    print("Example 8: Fast Transcription Mode")
    print("=" * 60)

    config = TranscriptionConfig(
        model="small",  # Smaller model = faster
        device="cuda",
        compute_type="float16",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.6,                # Skip marginal speech
            "min_speech_duration_ms": 500,   # Ignore very short segments
            "min_silence_duration_ms": 1500, # Faster segmentation
            "speech_pad_ms": 200,            # Minimal padding
        },
        num_workers=2,  # Parallel processing
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files in FAST mode")
    print("Use case: Quick drafts, rough transcripts, batch processing")


def example_vad_with_multi_gpu():
    """Example 9: Combine VAD with multi-GPU parallelization."""
    print("\n" + "=" * 60)
    print("Example 9: VAD + Multi-GPU (Maximum Performance)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        num_workers=2,          # Multi-GPU parallelization
        vad_filter=True,        # VAD for quality + speed
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 2000,
        },
        output_formats={"json", "srt", "vtt"},
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with VAD + Multi-GPU")
    print("Result: Fast + High Quality transcriptions")


def example_error_handling():
    """Example 10: VAD parameter validation errors."""
    print("\n" + "=" * 60)
    print("Example 10: Parameter Validation Errors")
    print("=" * 60)

    # Example 1: Invalid threshold
    try:
        validate_vad_parameters({"threshold": 1.5})
    except ValueError as e:
        print(f"❌ Error: {e}")

    # Example 2: Invalid window size (not power of 2)
    try:
        validate_vad_parameters({"window_size_samples": 1000})
    except ValueError as e:
        print(f"❌ Error: {e}")

    # Example 3: Negative duration
    try:
        validate_vad_parameters({"min_speech_duration_ms": -100})
    except ValueError as e:
        print(f"❌ Error: {e}")

    print("\n✅ All validation errors caught correctly")


def main():
    """Run selected examples."""
    print("\n" + "=" * 60)
    print("ytpipe Voice Activity Detection (VAD) Examples")
    print("=" * 60)

    # Comment/uncomment examples you want to run

    # Basics
    # example_default_vad()
    example_inspect_defaults()

    # Presets for different use cases
    # example_aggressive_vad()
    # example_sensitive_vad()
    # example_podcast_optimized()
    # example_fast_transcription()

    # Advanced
    # example_disable_vad()
    # example_validate_custom_params()
    # example_vad_with_multi_gpu()

    # Error handling
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
