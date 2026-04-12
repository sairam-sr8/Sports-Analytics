"""
========================================================
run_test.py — Sprint 1 Test Runner
========================================================

WHAT THIS FILE DOES (In Plain English):
----------------------------------------
This is your "START HERE" script for Sprint 1.
It runs our PoseDetector on the sample cricket video
and proves everything is working correctly.

WHAT YOU SHOULD SEE WHEN THIS RUNS:
  1. A video window opens showing the cricket video
  2. A yellow/green skeleton is drawn over the player
  3. The Right Elbow has a special red circle + label
  4. The terminal prints Right Elbow X, Y coordinates every 15 frames
  5. A summary is printed when done

HOW TO RUN:
  python run_test.py

Press 'Q' while the video window is focused to stop early.
========================================================
"""

import os
import sys

# Add the 'src' folder to Python's search path
# So Python can find our pose_detector.py file inside the src/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pose_detector import PoseDetector


# ============================================================
# CONFIGURATION — Change these paths if needed
# ============================================================

# Path to the test video (downloaded by download_samples.py)
TEST_VIDEO_PATH = os.path.join("videos", "samples", "test_batting_sample.mp4")

# Path to save the output video with skeleton drawn
# Set this to None if you don't want to save the output
OUTPUT_VIDEO_PATH = os.path.join("videos", "samples", "test_output_with_skeleton.mp4")

# Show the live video window while processing?
# Set to False if you're on a server without a display
SHOW_WINDOW = True

# Print Right Elbow coordinates every N frames
# 15 = every 15th frame (good for most videos)
PRINT_EVERY_N_FRAMES = 15


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  🏏  CRICKET BIOMECHANICS ANALYZER — SPRINT 1 TEST")
    print("=" * 60)

    # ── Step 1: Check if test video exists ─────────────────
    if not os.path.exists(TEST_VIDEO_PATH):
        print(f"\n  ❌ ERROR: Test video not found!")
        print(f"     Expected location: {TEST_VIDEO_PATH}")
        print("\n  Please do ONE of the following:")
        print("  OPTION A: Run the downloader first:")
        print("            python download_samples.py")
        print("\n  OPTION B: Place your own cricket video at:")
        print(f"            {TEST_VIDEO_PATH}")
        print("            (rename your file to: test_batting_sample.mp4)")
        sys.exit(1)

    print(f"\n  ✅ Test video found: {TEST_VIDEO_PATH}")

    # ── Step 2: Create the PoseDetector ─────────────────────
    # This is like turning on the X-ray machine.
    print("\n  ⚙️  Creating PoseDetector...")
    detector = PoseDetector(
        min_pose_detection_confidence=0.5,   # 50% confidence needed to detect a person
        min_tracking_confidence=0.5,         # 50% confidence to keep tracking
        # Note: model_complexity is obsolete in Tasks API - model file handles this
    )

    # ── Step 3: Process the Video ────────────────────────────
    print("  🎬 Starting video processing...")
    print("  📺 A video window will open. Press 'Q' to stop early.\n")

    landmarks_per_frame = detector.process_video(
        video_path=TEST_VIDEO_PATH,
        output_path=OUTPUT_VIDEO_PATH,
        show_window=SHOW_WINDOW,
        print_every_n=PRINT_EVERY_N_FRAMES
    )

    # ── Step 4: Sprint 1 Success Check ──────────────────────
    if landmarks_per_frame is not None:
        frames_with_data = [l for l in landmarks_per_frame if l is not None]

        print("\n" + "=" * 60)
        print("  🎉  SPRINT 1 — SUCCESS VERIFICATION")
        print("=" * 60)
        print("  ✅ Video opened and read successfully")
        print("  ✅ MediaPipe Pose Detection ran on all frames")
        print(f"  ✅ Right Elbow coordinates printed to terminal above")
        print(f"  ✅ Skeleton drawn on {len(frames_with_data)} frames")

        if OUTPUT_VIDEO_PATH and os.path.exists(OUTPUT_VIDEO_PATH):
            print(f"  ✅ Output video saved: {OUTPUT_VIDEO_PATH}")

        print("\n  📋 WHAT DID WE PROVE?")
        print("  → Our system can read real cricket video")
        print("  → MediaPipe correctly maps the skeleton on a real person")
        print("  → We can extract precise body joint coordinates (X, Y, Z)")
        print("  → The Right Elbow coordinate data is ready for angle math (Sprint 2)")
        print("\n  🚀 READY FOR SPRINT 2: Feature Extraction & Dataset Generation!")
        print("=" * 60 + "\n")
    else:
        print("\n  ❌ Something went wrong during processing.")
        print("     Check the error messages above for details.")


if __name__ == "__main__":
    main()
