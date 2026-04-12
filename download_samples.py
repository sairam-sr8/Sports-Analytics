"""
========================================================
download_samples.py  Sample Video Downloader
========================================================

WHAT THIS FILE DOES (In Plain English):
----------------------------------------
This script automatically downloads cricket batting videos
from YouTube so you can test the Pose Detector without
needing to find videos yourself.

It downloads:
  - 3 "GOOD technique" videos  videos/good/
  - 3 "BAD technique" videos   videos/bad/
  - 1 quick test video         videos/samples/

HOW TO RUN THIS:
  python download_samples.py

REQUIREMENTS:
  pip install yt-dlp must have been run first.

NOTE: This only downloads publicly available educational videos.
      All videos are coaching/tutorial content.
========================================================
"""

import subprocess
import sys
import os


def install_ytdlp():
    """Make sure yt-dlp is installed before we try to use it."""
    try:
        import yt_dlp
        print("   yt-dlp is already installed.")
        return True
    except ImportError:
        print("    Installing yt-dlp...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])
        print("   yt-dlp installed successfully!")
        return True


def download_video(url, output_folder, filename):
    """
    Download a single video from YouTube.

    Args:
        url (str): YouTube video URL
        output_folder (str): Where to save the video
        filename (str): What to name the downloaded file (without extension)
    """
    import yt_dlp

    os.makedirs(output_folder, exist_ok=True)
    output_template = os.path.join(output_folder, f"{filename}.%(ext)s")

    # yt-dlp options
    # We want:
    #   - Best quality video that is also mp4 (common format)
    #   - Maximum 480p resolution (keeps file size small for testing)
    #   - No playlist (single video only)
    ydl_opts = {
        'format':           'bestvideo[ext=mp4][height<=480]+bestaudio[ext=m4a]/best[ext=mp4][height<=480]/best',
        'outtmpl':          output_template,
        'noplaylist':       True,
        'quiet':            False,
        'no_warnings':      False,
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"\n   Downloading: {filename}")
            print(f"     URL: {url}")
            ydl.download([url])
            print(f"   Saved to: {output_folder}/{filename}.mp4")
            return True
    except Exception as e:
        print(f"    Download failed for {filename}: {e}")
        print("     This might be a network issue or the video was removed.")
        print("     Skipping this video and continuing...")
        return False


def main():
    """
    Main function that downloads all sample videos.
    """
    print("\n" + "=" * 60)
    print("    CRICKET SAMPLE VIDEO DOWNLOADER")
    print("=" * 60)
    print("\n  This will download sample cricket batting videos")
    print("  for training and testing our Biomechanics Analyzer.")
    print("  Download size: approximately 50-150 MB total.")
    print("\n  Note: Videos are publicly available coaching content.")
    print("=" * 60)

    install_ytdlp()

    # ===========================================================
    # GOOD TECHNIQUE VIDEOS
    # These are professional/coached batting techniques.
    # Sources: Official coaching channels, professional matches
    # ===========================================================
    good_videos = [
        {
            "url":  "https://www.youtube.com/watch?v=LquB7KN68h4",
            "name": "good_cover_drive_coaching",
            "desc": "Cover Drive coaching tutorial  correct technique"
        },
        {
            "url":  "https://www.youtube.com/watch?v=7sMrX8SZ3MQ",
            "name": "good_batting_basics_bcci",
            "desc": "BCCI official batting fundamentals coaching"
        },
        {
            "url":  "https://www.youtube.com/watch?v=Jv4FNOJWJOg",
            "name": "good_straight_drive",
            "desc": "Straight drive batting coaching  professional form"
        },
    ]

    # ===========================================================
    # BAD TECHNIQUE VIDEOS
    # These show common beginner mistakes in batting technique.
    # Sources: Common mistake analysis channels
    # ===========================================================
    bad_videos = [
        {
            "url":  "https://www.youtube.com/watch?v=nmfF5sEbVmY",
            "name": "bad_common_batting_mistakes",
            "desc": "Common cricket batting mistakes  beginners"
        },
        {
            "url":  "https://www.youtube.com/watch?v=b8FQzpxJy8A",
            "name": "bad_incorrect_stance",
            "desc": "Incorrect batting stance and grip analysis"
        },
    ]

    # ===========================================================
    # QUICK TEST VIDEO (just 1, short, to verify everything works)
    # ===========================================================
    test_videos = [
        {
            "url":  "https://www.youtube.com/watch?v=LquB7KN68h4",
            "name": "test_batting_sample",
            "desc": "Quick test video for Sprint 1 pose detection"
        },
    ]

    #  Download GOOD Videos 
    print(f"\n\n   DOWNLOADING 'GOOD TECHNIQUE' VIDEOS ")
    good_count = 0
    for video in good_videos:
        print(f"\n   {video['desc']}")
        success = download_video(video["url"], "videos/good", video["name"])
        if success:
            good_count += 1

    #  Download BAD Videos 
    print(f"\n\n   DOWNLOADING 'BAD TECHNIQUE' VIDEOS ")
    bad_count = 0
    for video in bad_videos:
        print(f"\n   {video['desc']}")
        success = download_video(video["url"], "videos/bad", video["name"])
        if success:
            bad_count += 1

    #  Download Test Video 
    print(f"\n\n   DOWNLOADING TEST VIDEO ")
    test_count = 0
    for video in test_videos:
        print(f"\n   {video['desc']}")
        success = download_video(video["url"], "videos/samples", video["name"])
        if success:
            test_count += 1

    #  Final Summary 
    print(f"\n\n{'='*60}")
    print(f"   DOWNLOAD COMPLETE  SUMMARY")
    print(f"{'='*60}")
    print(f"  Good technique videos downloaded : {good_count}/{len(good_videos)}")
    print(f"  Bad technique videos downloaded  : {bad_count}/{len(bad_videos)}")
    print(f"  Test videos downloaded           : {test_count}/{len(test_videos)}")
    print(f"{'='*60}")

    if test_count > 0:
        print("\n   READY FOR SPRINT 1!")
        print("  Now run: python run_test.py")
        print("      This will show the Pose Detector on your test video.\n")
    else:
        print("\n    Test video failed to download.")
        print("  Please place any cricket .mp4 video in: videos/samples/")
        print("  Rename it to: test_batting_sample.mp4")
        print("  Then run: python run_test.py\n")


if __name__ == "__main__":
    main()

