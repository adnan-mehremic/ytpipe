"""
LLM-based post-processing for transcriptions.

This module provides functions for:
- Proofreading/correcting transcriptions (grammar, misheard words)
- Translating transcriptions while preserving timestamp format
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _load_env():
    """Load environment variables from .env file if it exists."""
    try:
        from dotenv import load_dotenv
        # Look for .env in current directory and parent directories
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded .env from {env_path}")
        else:
            # Try parent directory
            env_path = Path.cwd().parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                logger.debug(f"Loaded .env from {env_path}")
            else:
                load_dotenv()  # Try default locations
    except ImportError:
        pass  # python-dotenv not installed, skip


# Load .env on module import
_load_env()


@dataclass
class LLMConfig:
    """Configuration for LLM post-processing."""

    api_key: Optional[str] = None
    model: str = "gpt-4o"
    base_url: Optional[str] = None  # For custom endpoints
    temperature: float = 0.3  # Lower for more consistent output
    max_tokens: int = 16000

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )


def _get_openai_client(config: LLMConfig):
    """Get OpenAI client with configuration."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "OpenAI package required for LLM post-processing. "
            "Install with: pip install openai"
        )

    return OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )


def proofread_text(
    text: str,
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
) -> str:
    """
    Proofread and correct transcription text using LLM.

    Fixes:
    - Grammar errors
    - Misheard/mistranscribed words
    - Punctuation
    - Formatting

    Parameters
    ----------
    text : str
        The transcription text to proofread.
    config : LLMConfig, optional
        LLM configuration. If None, uses defaults with env API key.
    source_language : str
        The language of the text (default: Bosnian).

    Returns
    -------
    str
        Corrected text.
    """
    if config is None:
        config = LLMConfig()

    client = _get_openai_client(config)

    system_prompt = f"""You are an expert proofreader and editor specializing in {source_language} language transcriptions from speech-to-text systems.

## Your Task
Correct transcription errors while preserving the speaker's original meaning and intent.

## Critical Rules
1. **Idioms & Expressions**: Recognize and correct common idioms that were mistranscribed phonetically.
   - Example: "noge mi se osjete" → "noge mi se odsjeku" (idiom meaning shock/surprise)
   - Example: "pasti na pamet" not "pasti na pametu"

2. **Phonetic Errors**: Speech-to-text often mishears similar-sounding words. Use context to determine the correct word.
   - Example: "osjete" vs "odsjeku" - choose based on meaning
   - Example: "čerku" → "kćerku" (daughter)

3. **Grammar & Spelling**: Fix grammatical errors, spelling mistakes, and punctuation.

4. **Preserve Structure**: Keep the same paragraph/line breaks. Do NOT merge or split lines.

5. **No Paraphrasing**: Fix errors but do NOT change the meaning or rephrase sentences.

6. **Regional Variants**: Accept valid Bosnian/Croatian/Serbian regional variations - don't "standardize" natural speech.

7. **Output Only**: Return ONLY the corrected text. No explanations, comments, or notes.

## Islamic Abbreviations (MANDATORY)
- Any mention of "džellešanuhu" or similar transcriptions of "جل جلاله" → replace with "dž.š."
- Any mention of "sallallahu alejhi we sellem", "alejhi salatu we selam", "alejhis-selam" or similar → replace with "s.a.v.s."

## Standard Introduction and Conclusion
- The introduction of each transcript should be: "U ime Allaha, hvala i zahvala pripada Allahu, salavat i selam neka je na Poslanika Muhammeda, s.a.v.s. Selam alejkum we rahmetullah we berekatuhu!"
- The conclusion/ending of each transcript should be: "Amin ja Rabbe-l-'alemin. Subhaneke Allahumme ve bihamdike. Ešhedu en la ilahe illa ente. Ve etubu ilejk."
- If the transcript contains a garbled/mistranscribed version of these phrases, replace them with the standard versions above.

## Common Speech-to-Text Errors to Watch For
- Mistranscribed idioms and expressions
- Wrong word boundaries (words split or merged incorrectly)
- Homophone confusion (similar sounding words)
- Missing or extra words
- Incorrect verb forms or cases"""

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content.strip()


def translate_text(
    text: str,
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
    target_language: str = "English",
) -> str:
    """
    Translate transcription text using LLM.

    Parameters
    ----------
    text : str
        The text to translate.
    config : LLMConfig, optional
        LLM configuration. If None, uses defaults with env API key.
    source_language : str
        Source language (default: Bosnian).
    target_language : str
        Target language (default: English).

    Returns
    -------
    str
        Translated text.
    """
    if config is None:
        config = LLMConfig()

    client = _get_openai_client(config)

    system_prompt = f"""You are a professional translator from {source_language} to {target_language}.

Your task is to translate the text:
1. Provide accurate, natural-sounding translation
2. Preserve the original meaning and tone
3. Keep the same paragraph/line structure as the original
4. Do NOT add any commentary or explanations - only return the translated text
5. Maintain any cultural context appropriately"""

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    return response.choices[0].message.content.strip()


def proofread_srt(
    srt_content: str,
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
) -> str:
    """
    Proofread SRT subtitle content while preserving timestamps.

    Parameters
    ----------
    srt_content : str
        The SRT file content.
    config : LLMConfig, optional
        LLM configuration.
    source_language : str
        The language of the subtitles.

    Returns
    -------
    str
        Corrected SRT content with timestamps preserved.
    """
    if config is None:
        config = LLMConfig()

    client = _get_openai_client(config)

    system_prompt = f"""You are an expert proofreader specializing in {source_language} subtitles from speech-to-text transcription.

## SRT Format Rules (CRITICAL)
- Keep ALL timestamp lines EXACTLY as they are (e.g., "00:00:06,320 --> 00:00:15,440")
- Keep ALL sequence numbers EXACTLY as they are
- Do NOT add, remove, or merge subtitle blocks
- Output must be a valid SRT file with the SAME number of blocks

## Proofreading Rules
1. **Idioms & Expressions**: Recognize and correct common idioms that were mistranscribed phonetically.
   - Example: "noge mi se osjete" → "noge mi se odsjeku" (idiom meaning shock/surprise)
   - Example: "pasti na pamet" not "pasti na pametu"

2. **Phonetic Errors**: Speech-to-text often mishears similar-sounding words. Use context to determine the correct word.
   - Example: "osjete" vs "odsjeku" - choose based on meaning
   - Example: "čerku" → "kćerku" (daughter)

3. **Grammar & Spelling**: Fix grammatical errors, spelling mistakes, and punctuation.

4. **No Paraphrasing**: Fix errors but do NOT change the meaning or rephrase sentences.

5. **Regional Variants**: Accept valid Bosnian/Croatian/Serbian regional variations.

6. **Output Only**: Return ONLY the corrected SRT content. No explanations or comments.

## Islamic Abbreviations (MANDATORY)
- Any mention of "džellešanuhu" or similar transcriptions of "جل جلاله" → replace with "dž.š."
- Any mention of "sallallahu alejhi we sellem", "alejhi salatu we selam", "alejhis-selam" or similar → replace with "s.a.v.s."

## Standard Introduction and Conclusion
- The introduction of each transcript should be: "U ime Allaha, hvala i zahvala pripada Allahu, salavat i selam neka je na Poslanika Muhammeda, s.a.v.s. Selam alejkum we rahmetullah we berekatuhu!"
- The conclusion/ending of each transcript should be: "Amin ja Rabbe-l-'alemin. Subhaneke Allahumme ve bihamdike. Ešhedu en la ilahe illa ente. Ve etubu ilejk."
- If the subtitle contains a garbled/mistranscribed version of these phrases, replace them with the standard versions above.

## Common Speech-to-Text Errors
- Mistranscribed idioms and expressions
- Homophone confusion (similar sounding words)
- Wrong word boundaries
- Missing or extra words"""

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": srt_content},
        ],
    )

    return response.choices[0].message.content.strip()


def translate_srt(
    srt_content: str,
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
    target_language: str = "English",
) -> str:
    """
    Translate SRT subtitle content while preserving timestamps.

    Parameters
    ----------
    srt_content : str
        The SRT file content.
    config : LLMConfig, optional
        LLM configuration.
    source_language : str
        Source language.
    target_language : str
        Target language.

    Returns
    -------
    str
        Translated SRT content with timestamps preserved.
    """
    if config is None:
        config = LLMConfig()

    client = _get_openai_client(config)

    system_prompt = f"""You are a professional subtitle translator from {source_language} to {target_language}.

Your task is to translate the subtitles while PRESERVING the exact SRT format:
1. Keep ALL timestamp lines exactly as they are (e.g., "00:00:06,320 --> 00:00:15,440")
2. Keep ALL sequence numbers exactly as they are
3. Translate ONLY the text content lines
4. Provide natural-sounding translations that fit the timing
5. Do NOT change the structure or add/remove subtitle blocks
6. Return the complete SRT file with translations

IMPORTANT: The output must be a valid SRT file with the same number of subtitle blocks."""

    response = client.chat.completions.create(
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": srt_content},
        ],
    )

    return response.choices[0].message.content.strip()


def process_file(
    input_path: Path,
    output_path: Optional[Path] = None,
    operation: str = "proofread",  # "proofread" or "translate"
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
    target_language: str = "English",
) -> Path:
    """
    Process a transcription file (TXT or SRT) with LLM.

    Parameters
    ----------
    input_path : Path
        Path to input file (.txt or .srt).
    output_path : Path, optional
        Path for output file. If None, creates alongside input with suffix.
    operation : str
        "proofread" or "translate".
    config : LLMConfig, optional
        LLM configuration.
    source_language : str
        Source language.
    target_language : str
        Target language (only for translate operation).

    Returns
    -------
    Path
        Path to the output file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    content = input_path.read_text(encoding="utf-8")
    is_srt = input_path.suffix.lower() == ".srt"

    # Determine output path
    if output_path is None:
        if operation == "proofread":
            suffix = "_proofread"
        else:
            suffix = f"_{target_language.lower()}"
        output_path = input_path.with_stem(input_path.stem + suffix)

    output_path = Path(output_path)

    logger.info(f"Processing {input_path.name} ({operation})...")

    # Process based on operation and file type
    if operation == "proofread":
        if is_srt:
            result = proofread_srt(content, config, source_language)
        else:
            result = proofread_text(content, config, source_language)
    elif operation == "translate":
        if is_srt:
            result = translate_srt(content, config, source_language, target_language)
        else:
            result = translate_text(content, config, source_language, target_language)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'proofread' or 'translate'.")

    # Write output
    output_path.write_text(result, encoding="utf-8")
    logger.info(f"Wrote {output_path.name}")

    return output_path


def process_transcript_directory(
    transcript_dir: Path,
    operation: str = "proofread",
    config: Optional[LLMConfig] = None,
    source_language: str = "Bosnian",
    target_language: str = "English",
    file_types: tuple[str, ...] = (".txt", ".srt"),
) -> list[Path]:
    """
    Process all transcription files in a directory.

    Parameters
    ----------
    transcript_dir : Path
        Directory containing transcription files.
    operation : str
        "proofread" or "translate".
    config : LLMConfig, optional
        LLM configuration.
    source_language : str
        Source language.
    target_language : str
        Target language.
    file_types : tuple
        File extensions to process.

    Returns
    -------
    list[Path]
        List of output file paths.
    """
    transcript_dir = Path(transcript_dir)
    output_files = []

    for file_type in file_types:
        for input_file in transcript_dir.glob(f"*{file_type}"):
            # Skip already processed files
            if "_proofread" in input_file.stem or "_english" in input_file.stem:
                continue

            try:
                output_path = process_file(
                    input_file,
                    operation=operation,
                    config=config,
                    source_language=source_language,
                    target_language=target_language,
                )
                output_files.append(output_path)
            except Exception as e:
                logger.error(f"Failed to process {input_file.name}: {e}")

    return output_files
