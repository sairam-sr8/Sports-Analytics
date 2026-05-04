"""
========================================================
dataset_builder.py — The CSV Generator (V2 — 22 Features)
========================================================

V2 UPGRADE:
  - Outputs 22 biomechanical features per frame (up from 4)
  - Adds swing phase label per frame
  - Adds shot type label per frame
  - Preserves backward compatibility with older video datasets

WHAT THIS FILE DOES:
--------------------
1. Scans videos/good/ and videos/bad/ folders.
2. For every video:
    a. Runs PoseDetector frame-by-frame.
    b. Runs FeatureExtractor (22-feature math engine).
    c. Runs PhaseDetector (swing phase per frame).
    d. Runs ShotClassifier (shot type per video).
3. Tags 'good' videos with label=1 and 'bad' with label=0.
4. Saves all data into one comprehensive CSV.

AUTHOR: Cricket Biomechanics Analyzer V2
"""

import os
import glob
import pandas as pd
import time
import sys

# Ensure src/ is on path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from pose_detector import PoseDetector
from feature_extractor import FeatureExtractor
from phase_detector import PhaseDetector, PHASE_NAMES
from shot_classifier import ShotClassifier, SHOT_NAMES


class DatasetBuilder:
    def __init__(self,
                 good_dir="videos/good",
                 bad_dir="videos/bad",
                 output_csv="data/cricket_biomechanics_dataset.csv"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.good_dir   = os.path.join(base_dir, good_dir)
        self.bad_dir    = os.path.join(base_dir, bad_dir)
        self.output_csv = os.path.join(base_dir, output_csv)

        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)

        self.pose_detector  = PoseDetector(
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smoothing_window=5
        )
        self.feature_extractor = FeatureExtractor()
        self.global_timestamp_ms = 0  # Strictly increasing timestamp for MediaPipe

    def _process_video(self, video_path: str, label: int) -> list:
        """
        Extracts 22 features + phase label from every valid frame of a video.

        Args:
            video_path (str): Path to the .mp4 file.
            label (int):      1 = good technique, 0 = bad technique.

        Returns:
            list[dict]: Dataset rows for this video.
        """
        import cv2

        print(f"    -> Analyzing: {os.path.basename(video_path)} "
              f"(Label: {'GOOD' if label==1 else 'BAD'})")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"       [WARNING] Could not open {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Per-video temporal engines (reset for each video)
        phase_detector  = PhaseDetector()
        shot_classifier = ShotClassifier()

        video_data   = []
        frame_counter = 0
        valid_frames  = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            self.global_timestamp_ms += 33

            # Step 1: Skeleton detection
            _, landmarks = self.pose_detector.process_frame(frame, self.global_timestamp_ms)

            # Step 2: 22-feature extraction
            features = self.feature_extractor.extract_features(landmarks)
            if features is None:
                continue

            # Step 3: Phase detection
            phase_id, phase_name, _ = phase_detector.detect_phase(features)

            # Step 4: Shot classification (uses sliding window internally)
            shot_id, shot_name, _ = shot_classifier.classify_frame(features)

            # Assemble the dataset row
            row = dict(features)
            row['video_name'] = os.path.basename(video_path)
            row['frame']      = frame_counter
            row['phase']      = phase_name
            row['shot_type']  = shot_name
            row['label']      = label

            video_data.append(row)
            valid_frames += 1

        cap.release()
        print(f"       Extracted {valid_frames} valid frames out of {total_frames} total.")
        return video_data

    def build(self) -> pd.DataFrame:
        """
        Main method: batch processes all good/bad videos → CSV dataset.
        """
        print("\n" + "="*60)
        print("  CRICKET BIOMECHANICS — DATASET BUILDER (V2 — 22 Features)")
        print("="*60)

        good_videos = glob.glob(os.path.join(self.good_dir, "*.mp4"))
        bad_videos  = glob.glob(os.path.join(self.bad_dir,  "*.mp4"))

        print(f"  Found {len(good_videos)} 'Good' videos.")
        print(f"  Found {len(bad_videos)} 'Bad' videos.")

        if not good_videos and not bad_videos:
            print("  ❌ ERROR: No videos found!")
            return None

        all_rows   = []
        start_time = time.time()

        print("\n  [ Phase 1/2 ] Processing GOOD Technique Videos...")
        for vid in good_videos:
            all_rows.extend(self._process_video(vid, label=1))

        print("\n  [ Phase 2/2 ] Processing BAD Technique Videos...")
        for vid in bad_videos:
            all_rows.extend(self._process_video(vid, label=0))

        print("\n" + "="*60)
        if not all_rows:
            print("  ❌ ERROR: Extraction failed. No usable frames found.")
            return None

        df = pd.DataFrame(all_rows)

        # Column ordering: metadata first, 22 features, phase/shot, label last
        meta_cols    = ['video_name', 'frame']
        feature_cols = self.feature_extractor.get_feature_names()
        extra_cols   = ['phase', 'shot_type', 'label']

        # Keep only columns that actually exist (graceful handling)
        ordered_cols = [c for c in meta_cols + feature_cols + extra_cols if c in df.columns]
        df = df[ordered_cols]

        df.to_csv(self.output_csv, index=False)

        elapsed = time.time() - start_time
        print(f"  ✅ DATASET GENERATION COMPLETE!")
        print(f"  Total Data Points (Frames) : {len(df)}")
        print(f"  Feature Columns            : {len(feature_cols)}")
        print(f"  Saved CSV to               : {self.output_csv}")
        print(f"  Time Taken                 : {elapsed:.1f} seconds")
        print("="*60 + "\n")

        self.pose_detector.close()
        return df
