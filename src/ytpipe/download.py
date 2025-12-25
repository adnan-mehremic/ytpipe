from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Sequence, List
import json
import logging
import os

import yt_dlp
from tqdm import tqdm

from .config import DownloadConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DownloadedItem:
    """
    Metadata about a single downloaded YouTube item (audio + info JSON).
    """

    id: str
    title: str | None
    uploader: str | None
    webpage_url: str | None
    duration: float | None
    upload_date: str | None
    audio_path: Path
    info_json_path: Path
    source: str = "youtube"


def _ensure_dirs(base_out: Path) -> dict[str, Path]:
    """
    Ensure required output directories exist under the base output directory.

    Returns a mapping of directory roles to paths, e.g. {"audio": Path(...)}.
    """
    audio_dir = base_out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    return {"audio": audio_dir}


def _flatten_entries(info: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """
    Recursively flatten a yt-dlp result object that may represent
    a playlist, channel, or nested playlists.

    Yields leaf entries (i.e. actual video dicts).
    """
    if not info:
        return

    entries = info.get("entries")
    if entries:
        for child in entries:
            if child:
                # Recurse into nested playlists/channels
                yield from _flatten_entries(child)
    else:
        # Leaf entry (typically a single video)
        yield info


def _to_manifest_row(entry: Dict[str, Any], audio_dir: Path) -> dict[str, Any]:
    """
    Convert a yt-dlp entry into a JSON-serializable manifest row.
    """
    vid = entry.get("id")
    title = entry.get("title")
    url = entry.get("webpage_url")
    uploader = entry.get("uploader")
    duration = entry.get("duration")
    upload_date = entry.get("upload_date")
    ext = "m4a"
    audio_path = audio_dir / f"{vid}.{ext}"
    info_path = audio_dir / f"{vid}.info.json"

    return {
        "id": vid,
        "title": title,
        "uploader": uploader,
        "webpage_url": url,
        "duration": duration,
        "upload_date": upload_date,
        "audio_path": str(audio_path),
        "info_json": str(info_path),
        "source": "youtube",
    }


class TqdmHook:
    """
    yt-dlp progress hook that feeds a tqdm progress bar.
    """

    def __init__(self, desc: str = "download") -> None:
        self.pbar: Optional[tqdm] = None
        self.total: Optional[int] = None
        self.desc = desc

    def __call__(self, d: Dict[str, Any]) -> None:
        status = d.get("status")

        if status == "downloading":
            total = (
                d.get("total_bytes")
                or d.get("total_bytes_estimate")
                or d.get("total_bytes_approx")
            )
            downloaded = d.get("downloaded_bytes", 0)

            # (re)create bar if needed
            if total and (self.pbar is None or self.total != total):
                if self.pbar:
                    self.pbar.close()
                self.total = int(total)
                self.pbar = tqdm(
                    total=self.total, unit="B", unit_scale=True, desc=self.desc
                )

            # update bar
            if self.pbar:
                self.pbar.n = int(downloaded)
                self.pbar.refresh()

        elif status == "finished":
            if self.pbar:
                if self.pbar.total:
                    self.pbar.n = self.pbar.total
                self.pbar.close()
                self.pbar = None

        elif status == "error":
            if self.pbar:
                self.pbar.close()
                self.pbar = None


def download_sources(
    sources: Sequence[str],
    config: DownloadConfig,
) -> list[DownloadedItem]:
    """
    Download audio from given YouTube sources (videos/playlists/channels).

    Parameters
    ----------
    sources:
        Iterable of YouTube URLs.
    config:
        Download configuration options.

    Returns
    -------
    list[DownloadedItem]
        Metadata for all items encountered in this run.
        (Items that were skipped by yt-dlp due to the download archive
        may still appear here depending on yt-dlp behavior.)
    """
    if not sources:
        logger.info("No sources provided, nothing to do.")
        return []

    config.out_dir.mkdir(parents=True, exist_ok=True)
    dirs = _ensure_dirs(config.out_dir)
    audio_dir = dirs["audio"]

    archive_path = config.out_dir / config.archive_filename
    manifest_path = config.out_dir / config.manifest_filename

    # Base yt-dlp options; derived from your original script.
    ydl_opts: Dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "noplaylist": False,
        "ignoreerrors": True,
        "quiet": True,  # tqdm + logging instead
        "writesubtitles": False,
        "writeinfojson": True,
        "concurrent_fragment_downloads": config.concurrent_fragments,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "0",
            }
        ],
    }

    if config.use_archive:
        ydl_opts["download_archive"] = str(archive_path)
    if config.ffmpeg_path is not None:
        ydl_opts["ffmpeg_location"] = str(config.ffmpeg_path)
    if config.max_videos is not None:
        ydl_opts["playlist_items"] = f"1-{config.max_videos}"
    if config.extractor_args:
        # Workaround for JS runtime warning / SABR issues:
        # --extractor-args "youtube:player_client=default"
        ydl_opts["extractor_args"] = config.extractor_args

    # User-specified overrides
    ydl_opts.update(config.ydl_extra_opts)

    hook = TqdmHook()
    if config.show_progress:
        ydl_opts["progress_hooks"] = [hook]

    items: list[DownloadedItem] = []
    total = 0

    # Manifest file handle (optional)
    manifest_file = None
    if config.write_manifest:
        manifest_file = manifest_path.open("a", encoding="utf-8")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for src in sources:
                logger.info("Processing source: %s", src)
                try:
                    info = ydl.extract_info(src, download=True)
                except Exception:
                    logger.exception("Failed to process source: %s", src)
                    continue

                for entry in _flatten_entries(info):
                    if config.max_videos is not None and total >= config.max_videos:
                        logger.info(
                            "Reached max_videos=%d limit. Stopping.",
                            config.max_videos,
                        )
                        logger.info(
                            "Downloaded/registered %d videos in this run.", total
                        )
                        return items

                    if config.show_progress and hook.pbar:
                        hook.pbar.set_description(
                            f"{entry.get('id', 'download')}"
                        )

                    row = _to_manifest_row(entry, audio_dir)
                    audio_path = Path(row["audio_path"])
                    info_json_path = Path(row["info_json"])

                    if manifest_file is not None:
                        manifest_file.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                        manifest_file.flush()
                        os.fsync(manifest_file.fileno())

                    item = DownloadedItem(
                        id=row["id"],
                        title=row.get("title"),
                        uploader=row.get("uploader"),
                        webpage_url=row.get("webpage_url"),
                        duration=row.get("duration"),
                        upload_date=row.get("upload_date"),
                        audio_path=audio_path,
                        info_json_path=info_json_path,
                        source=row.get("source", "youtube"),
                    )
                    items.append(item)

                    total += 1
                    logger.info("Saved: %s | %s", item.id, item.title)
    finally:
        if manifest_file is not None:
            manifest_file.close()

    logger.info("Done. Downloaded/registered %d videos.", total)
    return items
