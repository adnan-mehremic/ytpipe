from pathlib import Path

from ytpipe import (
    DownloadConfig,
    TranscriptionConfig,
    download_sources,
    transcribe_directory,
    setup_logging,
)

def main() -> None:
    setup_logging(verbose=True)

    # 1) Download at most N videos from a playlist
    playlist_url = "https://www.youtube.com/playlist?list=PL123..."
    raw_dir = Path("data/raw")

    dl_config = DownloadConfig(
        out_dir=raw_dir,
        max_videos=10,
        use_archive=True,
        show_progress=True,
    )

    downloaded_items = download_sources(
        sources=[playlist_url],
        config=dl_config,
    )

    print(f"Downloaded/registered {len(downloaded_items)} item(s).")

    # 2) Transcribe them
    audio_dir = raw_dir / "audio"
    transcripts_dir = Path("data/transcripts")

    tr_config = TranscriptionConfig(
        model="medium",
        device="auto",
        compute_type="auto",
        beam_size=5,
        vad_filter=True,
        language="en",
        skip_existing=True,
    )

    results = transcribe_directory(
        audio_dir=audio_dir,
        out_dir=transcripts_dir,
        config=tr_config,
    )

    # 3) Print paths of transcript JSON files
    for r in results:
        print(f"Transcript JSON: {r.json_path}")

if __name__ == "__main__":
    main()
