"""
Comprehensive tests for Voice Activity Detection (VAD) functionality.

Tests cover VAD parameter validation, configuration, and integration with
transcription workflow.
"""

from __future__ import annotations

import pytest

from ytpipe.config import (
    TranscriptionConfig,
    get_default_vad_parameters,
    validate_vad_parameters,
)


# =============================================================================
# VAD PARAMETER TESTS
# =============================================================================


class TestGetDefaultVADParameters:
    """Tests for get_default_vad_parameters function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        params = get_default_vad_parameters()
        assert isinstance(params, dict)

    def test_contains_all_required_keys(self):
        """Should contain all required VAD parameter keys."""
        params = get_default_vad_parameters()

        required_keys = {
            "threshold",
            "min_speech_duration_ms",
            "max_speech_duration_s",
            "min_silence_duration_ms",
            "window_size_samples",
            "speech_pad_ms",
        }

        assert set(params.keys()) == required_keys

    def test_threshold_is_float(self):
        """Threshold should be a float between 0 and 1."""
        params = get_default_vad_parameters()
        assert isinstance(params["threshold"], (int, float))
        assert 0 <= params["threshold"] <= 1

    def test_threshold_default_value(self):
        """Threshold should default to 0.5 (balanced)."""
        params = get_default_vad_parameters()
        assert params["threshold"] == 0.5

    def test_min_speech_duration_is_positive(self):
        """min_speech_duration_ms should be positive."""
        params = get_default_vad_parameters()
        assert params["min_speech_duration_ms"] > 0

    def test_min_silence_duration_is_non_negative(self):
        """min_silence_duration_ms should be non-negative."""
        params = get_default_vad_parameters()
        assert params["min_silence_duration_ms"] >= 0

    def test_window_size_is_power_of_two(self):
        """window_size_samples should be a power of 2."""
        params = get_default_vad_parameters()
        window_size = params["window_size_samples"]

        # Check if power of 2
        assert window_size > 0
        assert (window_size & (window_size - 1)) == 0

    def test_speech_pad_is_non_negative(self):
        """speech_pad_ms should be non-negative."""
        params = get_default_vad_parameters()
        assert params["speech_pad_ms"] >= 0


class TestValidateVADParameters:
    """Tests for validate_vad_parameters function."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        validated = validate_vad_parameters({})
        assert isinstance(validated, dict)

    def test_empty_dict_returns_defaults(self):
        """Empty dict should return default parameters."""
        validated = validate_vad_parameters({})
        defaults = get_default_vad_parameters()

        assert validated == defaults

    def test_none_input_returns_defaults(self):
        """None input should return default parameters."""
        validated = validate_vad_parameters(None)
        defaults = get_default_vad_parameters()

        assert validated == defaults

    def test_partial_params_merged_with_defaults(self):
        """Partial parameters should be merged with defaults."""
        user_params = {"threshold": 0.3}
        validated = validate_vad_parameters(user_params)

        assert validated["threshold"] == 0.3
        assert validated["min_speech_duration_ms"] == 250  # Default
        assert validated["min_silence_duration_ms"] == 2000  # Default

    def test_all_custom_params_preserved(self):
        """All custom parameters should be preserved."""
        custom_params = {
            "threshold": 0.7,
            "min_speech_duration_ms": 300,
            "max_speech_duration_s": 30.0,
            "min_silence_duration_ms": 1500,
            "window_size_samples": 512,
            "speech_pad_ms": 200,
        }

        validated = validate_vad_parameters(custom_params)

        assert validated == custom_params

    def test_threshold_validation_min_boundary(self):
        """Threshold 0 (minimum) should be valid."""
        validated = validate_vad_parameters({"threshold": 0.0})
        assert validated["threshold"] == 0.0

    def test_threshold_validation_max_boundary(self):
        """Threshold 1 (maximum) should be valid."""
        validated = validate_vad_parameters({"threshold": 1.0})
        assert validated["threshold"] == 1.0

    def test_threshold_below_zero_raises_error(self):
        """Threshold below 0 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            validate_vad_parameters({"threshold": -0.1})

    def test_threshold_above_one_raises_error(self):
        """Threshold above 1 should raise ValueError."""
        with pytest.raises(ValueError, match="threshold must be between 0 and 1"):
            validate_vad_parameters({"threshold": 1.5})

    def test_min_speech_duration_zero_raises_error(self):
        """min_speech_duration_ms of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="min_speech_duration_ms must be positive"):
            validate_vad_parameters({"min_speech_duration_ms": 0})

    def test_min_speech_duration_negative_raises_error(self):
        """Negative min_speech_duration_ms should raise ValueError."""
        with pytest.raises(ValueError, match="min_speech_duration_ms must be positive"):
            validate_vad_parameters({"min_speech_duration_ms": -100})

    def test_max_speech_duration_inf_is_valid(self):
        """max_speech_duration_s of infinity should be valid."""
        validated = validate_vad_parameters({"max_speech_duration_s": float("inf")})
        assert validated["max_speech_duration_s"] == float("inf")

    def test_max_speech_duration_positive_is_valid(self):
        """Positive max_speech_duration_s should be valid."""
        validated = validate_vad_parameters({"max_speech_duration_s": 30.0})
        assert validated["max_speech_duration_s"] == 30.0

    def test_max_speech_duration_zero_raises_error(self):
        """max_speech_duration_s of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="max_speech_duration_s must be positive"):
            validate_vad_parameters({"max_speech_duration_s": 0.0})

    def test_max_speech_duration_negative_raises_error(self):
        """Negative max_speech_duration_s should raise ValueError."""
        with pytest.raises(ValueError, match="max_speech_duration_s must be positive"):
            validate_vad_parameters({"max_speech_duration_s": -10.0})

    def test_min_silence_duration_zero_is_valid(self):
        """min_silence_duration_ms of 0 should be valid."""
        validated = validate_vad_parameters({"min_silence_duration_ms": 0})
        assert validated["min_silence_duration_ms"] == 0

    def test_min_silence_duration_negative_raises_error(self):
        """Negative min_silence_duration_ms should raise ValueError."""
        with pytest.raises(ValueError, match="min_silence_duration_ms must be non-negative"):
            validate_vad_parameters({"min_silence_duration_ms": -100})

    def test_window_size_power_of_two_valid(self):
        """Valid power of 2 for window_size_samples should work."""
        for size in [256, 512, 1024, 2048]:
            validated = validate_vad_parameters({"window_size_samples": size})
            assert validated["window_size_samples"] == size

    def test_window_size_not_power_of_two_raises_error(self):
        """Non-power-of-2 window_size_samples should raise ValueError."""
        with pytest.raises(ValueError, match="window_size_samples must be a positive power of 2"):
            validate_vad_parameters({"window_size_samples": 1000})

    def test_window_size_zero_raises_error(self):
        """window_size_samples of 0 should raise ValueError."""
        with pytest.raises(ValueError, match="window_size_samples must be a positive power of 2"):
            validate_vad_parameters({"window_size_samples": 0})

    def test_window_size_negative_raises_error(self):
        """Negative window_size_samples should raise ValueError."""
        with pytest.raises(ValueError, match="window_size_samples must be a positive power of 2"):
            validate_vad_parameters({"window_size_samples": -512})

    def test_speech_pad_zero_is_valid(self):
        """speech_pad_ms of 0 should be valid."""
        validated = validate_vad_parameters({"speech_pad_ms": 0})
        assert validated["speech_pad_ms"] == 0

    def test_speech_pad_positive_is_valid(self):
        """Positive speech_pad_ms should be valid."""
        validated = validate_vad_parameters({"speech_pad_ms": 500})
        assert validated["speech_pad_ms"] == 500

    def test_speech_pad_negative_raises_error(self):
        """Negative speech_pad_ms should raise ValueError."""
        with pytest.raises(ValueError, match="speech_pad_ms must be non-negative"):
            validate_vad_parameters({"speech_pad_ms": -100})


# =============================================================================
# TRANSCRIPTION CONFIG INTEGRATION TESTS
# =============================================================================


class TestTranscriptionConfigVAD:
    """Tests for VAD integration in TranscriptionConfig."""

    def test_vad_enabled_by_default(self):
        """VAD should be enabled by default (recommended)."""
        config = TranscriptionConfig()
        assert config.vad_filter is True

    def test_vad_can_be_disabled(self):
        """VAD should be able to be disabled."""
        config = TranscriptionConfig(vad_filter=False)
        assert config.vad_filter is False

    def test_vad_parameters_default_to_none(self):
        """vad_parameters should default to None (use library defaults)."""
        config = TranscriptionConfig()
        assert config.vad_parameters is None

    def test_vad_parameters_can_be_set(self):
        """vad_parameters should accept custom dict."""
        custom_params = {
            "threshold": 0.3,
            "min_speech_duration_ms": 300,
        }

        config = TranscriptionConfig(vad_parameters=custom_params)
        assert config.vad_parameters == custom_params

    def test_vad_parameters_empty_dict(self):
        """vad_parameters should accept empty dict."""
        config = TranscriptionConfig(vad_parameters={})
        assert config.vad_parameters == {}

    def test_config_with_all_vad_options(self):
        """Config should support all VAD options."""
        custom_params = {
            "threshold": 0.6,
            "min_speech_duration_ms": 200,
            "max_speech_duration_s": 60.0,
            "min_silence_duration_ms": 1000,
            "window_size_samples": 2048,
            "speech_pad_ms": 300,
        }

        config = TranscriptionConfig(
            model="large-v3",
            vad_filter=True,
            vad_parameters=custom_params,
        )

        assert config.vad_filter is True
        assert config.vad_parameters == custom_params


# =============================================================================
# VAD USE CASE TESTS
# =============================================================================


class TestVADUseCases:
    """Tests for common VAD use cases and presets."""

    def test_aggressive_vad_for_noisy_audio(self):
        """Aggressive VAD settings for very noisy audio."""
        # Higher threshold = less sensitive = stricter speech detection
        aggressive_params = {
            "threshold": 0.7,  # Very strict
            "min_speech_duration_ms": 500,  # Longer minimum
            "min_silence_duration_ms": 1000,  # Shorter silence tolerance
        }

        validated = validate_vad_parameters(aggressive_params)
        assert validated["threshold"] == 0.7
        assert validated["min_speech_duration_ms"] == 500

    def test_sensitive_vad_for_quiet_speech(self):
        """Sensitive VAD settings for quiet or whispering speech."""
        # Lower threshold = more sensitive = catches more speech
        sensitive_params = {
            "threshold": 0.3,  # Very sensitive
            "min_speech_duration_ms": 100,  # Catch short utterances
            "min_silence_duration_ms": 3000,  # Allow longer pauses
        }

        validated = validate_vad_parameters(sensitive_params)
        assert validated["threshold"] == 0.3
        assert validated["min_speech_duration_ms"] == 100

    def test_podcast_optimized_vad(self):
        """VAD settings optimized for podcast transcription."""
        podcast_params = {
            "threshold": 0.5,  # Balanced
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": 120.0,  # Long monologues
            "min_silence_duration_ms": 2000,  # Natural pauses
            "speech_pad_ms": 500,  # Don't cut off words
        }

        validated = validate_vad_parameters(podcast_params)
        assert validated["max_speech_duration_s"] == 120.0
        assert validated["speech_pad_ms"] == 500

    def test_fast_transcription_vad(self):
        """VAD settings optimized for speed over accuracy."""
        fast_params = {
            "threshold": 0.6,  # Skip marginal speech
            "min_speech_duration_ms": 500,  # Ignore very short segments
            "min_silence_duration_ms": 1500,  # Faster segmentation
        }

        validated = validate_vad_parameters(fast_params)
        assert validated["threshold"] == 0.6


# =============================================================================
# INTEGRATION WITH EXISTING TESTS
# =============================================================================


class TestVADBackwardsCompatibility:
    """Tests to ensure VAD changes don't break existing functionality."""

    def test_existing_config_still_works(self):
        """Existing config creation should still work."""
        config = TranscriptionConfig(
            model="medium",
            device="cpu",
            vad_filter=False,  # Explicit disable
        )

        assert config.model == "medium"
        assert config.device == "cpu"
        assert config.vad_filter is False
        assert config.vad_parameters is None

    def test_config_without_vad_parameters_works(self):
        """Config without vad_parameters should use None (defaults)."""
        config = TranscriptionConfig(
            model="large",
            vad_filter=True,
            # No vad_parameters specified
        )

        assert config.vad_filter is True
        assert config.vad_parameters is None
