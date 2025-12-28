from pathlib import Path

from ytpipe.config import TranscriptionConfig, get_default_vad_parameters, validate_vad_parameters
from ytpipe.transcribe import transcribe_directory


def example_default_vad():
    print("=" * 60)
    print("Example 1: Default VAD Configuration")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files with default VAD")
    print("Benefits: ~20-30% faster, fewer hallucinations, better timestamps")


def example_inspect_defaults():
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
    print("\n" + "=" * 60)
    print("Example 3: Aggressive VAD (Noisy Audio)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.7,
            "min_speech_duration_ms": 500,
            "min_silence_duration_ms": 1000,
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
    print("\n" + "=" * 60)
    print("Example 4: Sensitive VAD (Quiet Speech)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "min_silence_duration_ms": 3000,
            "speech_pad_ms": 500,
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
    print("\n" + "=" * 60)
    print("Example 5: Podcast-Optimized VAD")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 120.0,
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 500,
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
    print("\n" + "=" * 60)
    print("Example 6: Disable VAD")
    print("=" * 60)

    config = TranscriptionConfig(
        model="medium",
        vad_filter=False,
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
    print("\n" + "=" * 60)
    print("Example 7: Validate Custom Parameters")
    print("=" * 60)

    custom_params = {
        "threshold": 0.6,
        "min_speech_duration_ms": 300,
    }

    validated = validate_vad_parameters(custom_params)

    print("\nCustom parameters:")
    for key, value in custom_params.items():
        print(f"  {key}: {value}")

    print("\nValidated (complete) parameters:")
    for key, value in validated.items():
        print(f"  {key}: {value}")


def example_fast_transcription():
    print("\n" + "=" * 60)
    print("Example 8: Fast Transcription Mode")
    print("=" * 60)

    config = TranscriptionConfig(
        model="small",
        device="cuda",
        compute_type="float16",
        vad_filter=True,
        vad_parameters={
            "threshold": 0.6,
            "min_speech_duration_ms": 500,
            "min_silence_duration_ms": 1500,
            "speech_pad_ms": 200,
        },
        num_workers=2,
    )

    results = transcribe_directory(
        audio_dir=Path("data/raw/audio"),
        out_dir=Path("data/transcripts"),
        config=config,
    )

    print(f"\nTranscribed {len(results)} files in FAST mode")
    print("Use case: Quick drafts, rough transcripts, batch processing")


def example_vad_with_multi_gpu():
    print("\n" + "=" * 60)
    print("Example 9: VAD + Multi-GPU (Maximum Performance)")
    print("=" * 60)

    config = TranscriptionConfig(
        model="large-v3",
        device="cuda",
        compute_type="float16",
        num_workers=2,
        vad_filter=True,
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
    print("\n" + "=" * 60)
    print("Example 10: Parameter Validation Errors")
    print("=" * 60)

    try:
        validate_vad_parameters({"threshold": 1.5})
    except ValueError as e:
        print(f"Error: {e}")

    try:
        validate_vad_parameters({"window_size_samples": 1000})
    except ValueError as e:
        print(f"Error: {e}")

    try:
        validate_vad_parameters({"min_speech_duration_ms": -100})
    except ValueError as e:
        print(f"Error: {e}")

    print("\nAll validation errors caught correctly")


def main():
    print("\n" + "=" * 60)
    print("ytpipe Voice Activity Detection (VAD) Examples")
    print("=" * 60)

    example_inspect_defaults()
    example_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
