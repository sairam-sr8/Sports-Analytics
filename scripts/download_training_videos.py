# -*- coding: utf-8 -*-
"""
download_training_videos.py
============================
Downloads cricket batting YouTube videos for training data.
Run from project root: python scripts/download_training_videos.py
"""

import subprocess, os, sys

# Force UTF-8 output on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Good Technique: Pro analysis + coaching masterclasses
GOOD_VIDEOS = [
    "https://www.youtube.com/watch?v=OVtEQ5XkwEM",   # SKY technique analysis
    "https://www.youtube.com/watch?v=38m6Oujjo90",   # Kohli slow motion
    "https://www.youtube.com/watch?v=Xb0tXrmRYKs",   # Kohli vs Rohit off drive
    "https://www.youtube.com/watch?v=kX4bIXYTTIw",   # Rohit technique analysis
    "https://www.youtube.com/watch?v=pV79Zxcu0Zo",   # Kohli & Rohit nets
    "https://www.youtube.com/watch?v=HHdXyc7TxC8",   # Steve Smith technique
    "https://www.youtube.com/watch?v=dwkSLxMbByI",   # Gary Palmer perfect technique
    "https://www.youtube.com/watch?v=9iAl2g2ZFS8",   # Perfect grip & stance
    "https://www.youtube.com/watch?v=CQeXXrZfsvE",   # Mastering the drive
    "https://www.youtube.com/watch?v=73Xv-y9MjZ8",   # Pull shot masterclass
    "https://www.youtube.com/watch?v=HQMZMc_bpg8",   # All cricket shots explained
]

# Bad / Mixed technique for negative training examples
BAD_VIDEOS = [
    "https://www.youtube.com/watch?v=c29lO_6dins",   # Line/length judgment drills
    "https://www.youtube.com/watch?v=Fj_WWSAJ2qw",   # All shots part 2 (mixed)
]

def download_videos(urls, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*60)
    print(f"  Downloading {len(urls)} {label.upper()} videos -> {output_dir}")
    print("="*60)

    success, failed = 0, 0
    for i, url in enumerate(urls, 1):
        print(f"\n  [{i}/{len(urls)}] {url}")
        try:
            result = subprocess.run([
                sys.executable, "-m", "yt_dlp",
                # Best single-file mp4 <= 720p (no ffmpeg needed for merge)
                "--format", "best[height<=720][ext=mp4]/best[height<=720]/best",
                "--output", os.path.join(output_dir, f"{label}_yt_%(autonumber)03d.%(ext)s"),
                "--no-playlist",
                "--socket-timeout", "30",
                "--retries", "3",
                url
            ], timeout=300)

            if result.returncode == 0:
                print(f"  [OK] Downloaded successfully")
                success += 1
            else:
                print(f"  [WARN] yt-dlp returned code {result.returncode}")
                failed += 1
        except subprocess.TimeoutExpired:
            print(f"  [SKIP] Timeout (300s)")
            failed += 1
        except Exception as e:
            print(f"  [ERR] {e}")
            failed += 1

    print(f"\n  Summary: {success} downloaded, {failed} failed")
    return success

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CRICKET BIOMECHANICS - VIDEO DOWNLOADER")
    print("="*60)

    total_good = download_videos(GOOD_VIDEOS, "videos/good", "good")
    total_bad  = download_videos(BAD_VIDEOS,  "videos/bad",  "bad")

    print("\n" + "="*60)
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Good videos: {total_good}")
    print(f"  Bad videos : {total_bad}")
    print(f"  Total: {total_good + total_bad}")
    print("="*60)
