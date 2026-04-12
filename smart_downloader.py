"""
========================================================
smart_downloader.py — Intelligent Cricket Video Downloader
========================================================

WHAT THIS FILE DOES:
--------------------
Downloads cricket batting videos from YouTube using SEARCH TERMS
(not fixed URLs that go dead). This means it always finds
fresh, available videos.

Downloads into two folders:
  videos/good/  ← Videos of correct batting technique
  videos/bad/   ← Videos of incorrect/beginner technique

HOW TO RUN:
  python smart_downloader.py

This will download approximately 10 good + 10 bad videos.
Total download size: ~200-500 MB depending on video lengths.
"""

import os
import sys

# Force UTF-8 output (prevents emoji crash on Windows)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def check_ytdlp():
    """Make sure yt-dlp is installed."""
    try:
        import yt_dlp
        return True
    except ImportError:
        print("Installing yt-dlp...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])
        return True


def download_by_search(search_query, output_folder, filename_prefix,
                       max_videos=3, max_duration_secs=300, min_duration_secs=15):
    """
    Search YouTube for a query and download the top matching videos.

    Args:
        search_query (str): YouTube search terms (like what you'd type in YouTube)
        output_folder (str): Where to save the downloaded videos
        filename_prefix (str): Prefix for the saved file names
        max_videos (int): Maximum number of videos to download
        max_duration_secs (int): Skip videos longer than this (300 = 5 minutes max)
        min_duration_secs (int): Skip videos shorter than this (15 = 15 seconds min)

    Returns:
        int: Number of videos successfully downloaded
    """
    import yt_dlp

    os.makedirs(output_folder, exist_ok=True)

    # Build the YouTube search URL
    # "ytsearch3:cricket batting" = search YouTube and give me top 3 results
    search_url = f"ytsearch{max_videos * 3}:{search_query}"

    output_template = os.path.join(output_folder, f"{filename_prefix}_%(autonumber)s.%(ext)s")

    ydl_opts = {
        # Video format: best quality mp4, max 480p (smaller file size)
        'format': 'best[ext=mp4][height<=480]/bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[height<=480]/best',

        # Output file naming
        'outtmpl': output_template,

        # Search & playlist settings
        'noplaylist': True,
        'playlistend': max_videos * 3,    # Search top N*3 results to find N valid ones

        # Duration filter (skip too-short or too-long videos)
        'match_filter': yt_dlp.utils.match_filter_func(
            f"duration > {min_duration_secs} & duration < {max_duration_secs}"
        ),

        # Only download max_videos
        'max_downloads': max_videos,

        # Output format
        'merge_output_format': 'mp4',

        # Quiet mode (less spam in terminal)
        'quiet': False,
        'no_warnings': True,

        # Don't re-download already downloaded files
        'skip_download': False,
    }

    downloaded_count = 0
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\n  Searching YouTube: '{search_query}'")
            print(f"  Target: {max_videos} videos, max {max_duration_secs//60} min each")
            ydl.download([search_url])
            # Count how many files were saved
            downloaded_count = len([
                f for f in os.listdir(output_folder)
                if f.startswith(filename_prefix) and f.endswith('.mp4')
            ])
    except yt_dlp.utils.MaxDownloadsReached:
        # This is normal - it means we got our max_videos
        downloaded_count = max_videos
        print(f"  [OK] Reached download limit of {max_videos} videos.")
    except Exception as e:
        print(f"  [WARNING] Some downloads failed: {str(e)[:100]}")

    print(f"  Saved to: {output_folder}/")
    return downloaded_count


def main():
    print("\n" + "=" * 65)
    print("  CRICKET BIOMECHANICS — SMART VIDEO DOWNLOADER")
    print("=" * 65)
    print("  This downloads cricket batting videos from YouTube")
    print("  using SEARCH (no fixed URLs that go dead).")
    print("=" * 65)

    check_ytdlp()

    # =======================================================
    # GOOD TECHNIQUE SEARCHES
    # These search terms find professional coaching videos
    # showing CORRECT batting technique
    # =======================================================
    good_searches = [
        {
            "query":  "cricket batting cover drive perfect technique tutorial coaching",
            "prefix": "good_cover_drive",
            "count":  3,
        },
        {
            "query":  "cricket batting stance footwork tutorial correct technique slow motion",
            "prefix": "good_batting_basics",
            "count":  3,
        },
        {
            "query":  "Virat Kohli batting technique analysis cricket coaching",
            "prefix": "good_professional",
            "count":  2,
        },
        {
            "query":  "cricket batting straight drive pull shot correct form tutorial",
            "prefix": "good_shot_types",
            "count":  2,
        },
    ]

    # =======================================================
    # BAD TECHNIQUE SEARCHES
    # These search terms find videos showing WRONG technique,
    # common mistakes, and what NOT to do
    # =======================================================
    bad_searches = [
        {
            "query":  "common cricket batting mistakes beginners wrong technique errors",
            "prefix": "bad_mistakes",
            "count":  3,
        },
        {
            "query":  "cricket batting analysis wrong stance posture incorrect form",
            "prefix": "bad_stance",
            "count":  3,
        },
        {
            "query":  "cricket batting technique errors what not to do coaching",
            "prefix": "bad_errors",
            "count":  2,
        },
        {
            "query":  "how to fix cricket batting mistakes improvement tips",
            "prefix": "bad_corrections",
            "count":  2,
        },
    ]

    # ── Download GOOD videos ─────────────────────────────────
    print(f"\n\n  ============================================")
    print(f"  DOWNLOADING GOOD TECHNIQUE VIDEOS")
    print(f"  ============================================")

    total_good = 0
    for search in good_searches:
        count = download_by_search(
            search_query=search["query"],
            output_folder="videos/good",
            filename_prefix=search["prefix"],
            max_videos=search["count"],
            max_duration_secs=480,   # Max 8 minutes
            min_duration_secs=20,    # Min 20 seconds
        )
        total_good += count
        print(f"  -> Downloaded {count} videos for: {search['prefix']}")

    # ── Download BAD videos ──────────────────────────────────
    print(f"\n\n  ============================================")
    print(f"  DOWNLOADING BAD TECHNIQUE VIDEOS")
    print(f"  ============================================")

    total_bad = 0
    for search in bad_searches:
        count = download_by_search(
            search_query=search["query"],
            output_folder="videos/bad",
            filename_prefix=search["prefix"],
            max_videos=search["count"],
            max_duration_secs=480,
            min_duration_secs=20,
        )
        total_bad += count
        print(f"  -> Downloaded {count} videos for: {search['prefix']}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*65}")
    print(f"  Good technique videos : {total_good}")
    print(f"  Bad technique videos  : {total_bad}")
    print(f"  Total                 : {total_good + total_bad}")
    print(f"{'='*65}")

    if total_good >= 5 and total_bad >= 5:
        print(f"\n  READY FOR SPRINT 2!")
        print(f"  Run: python run_dataset_builder.py")
        print(f"  This will generate cricket_biomechanics_dataset.csv\n")
    else:
        print(f"\n  NOTE: We have {total_good} good + {total_bad} bad videos.")
        print(f"  Minimum recommended: 5 good + 5 bad.")
        print(f"  You can add more manually to videos/good/ and videos/bad/")
        print(f"  Sprint 2 can still run with fewer videos.\n")


if __name__ == "__main__":
    main()
