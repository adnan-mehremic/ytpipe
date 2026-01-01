from __future__ import annotations

import os
import sys


def _load_nvidia_libs():
    """
    Locate and register NVIDIA libraries installed via pip.

    On Windows, adds cuDNN and cuBLAS library paths to the system PATH
    so that the C++ backend (ctranslate2) can find the required DLLs.
    """
    if sys.platform != "win32":
        return

    try:
        import nvidia.cudnn
        import nvidia.cublas

        libs_paths = []

        for module in [nvidia.cudnn, nvidia.cublas]:
            if hasattr(module, "__path__") and len(module.__path__) > 0:
                base_dir = list(module.__path__)[0]
            elif hasattr(module, "__file__") and module.__file__ is not None:
                base_dir = os.path.dirname(module.__file__)
            else:
                continue

            libs_paths.append(os.path.join(base_dir, "bin"))
            libs_paths.append(os.path.join(base_dir, "lib"))

        for path in libs_paths:
            if os.path.isdir(path):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass

                if path not in os.environ["PATH"]:
                    os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]

    except ImportError:
        pass
    except Exception:
        pass


_load_nvidia_libs()
from .config import (
    DownloadConfig,
    TranscriptionConfig,
    setup_logging,
    SUPPORTED_OUTPUT_FORMATS,
    DEFAULT_OUTPUT_FORMATS,
)
from .download import DownloadedItem, download_sources
from .transcribe import (
    Segment,
    TranscriptionResult,
    transcribe_directory,
    resolve_device_and_compute,
    segments_to_srt,
    segments_to_vtt,
)

__all__ = [
    "DownloadConfig",
    "TranscriptionConfig",
    "DownloadedItem",
    "download_sources",
    "Segment",
    "TranscriptionResult",
    "transcribe_directory",
    "resolve_device_and_compute",
    "segments_to_srt",
    "segments_to_vtt",
    "setup_logging",
    "SUPPORTED_OUTPUT_FORMATS",
    "DEFAULT_OUTPUT_FORMATS",
]

__version__ = "0.1.7"