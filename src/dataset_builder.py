"""
========================================================
dataset_builder.py — The CSV Generator
========================================================

WHAT THIS FILE DOES:
--------------------
Orchestrates Sprint 2.
1. Scans the videos/good/ folder and videos/bad/ folder.
2. For every video:
    a. Runs the PoseDetector (digital skeleton) frame-by-frame.
    b. Runs the FeatureExtractor (angle math) on the skeleton.
3. Tags 'good' videos with label=1 and 'bad' with label=0.
4. Saves all hundreds/thousands of frames into one CSV file.

We will use this CSV file in Sprint 3 to train our AI model.
"""

import os
import glob
import pandas as pd
import time
from pose_detector import PoseDetector
from feature_extractor import FeatureExtractor

class DatasetBuilder:
    def __init__(self, good_dir="videos/good", bad_dir="videos/bad", output_csv="data/cricket_biomechanics_dataset.csv"):
        self.good_dir = good_dir
        self.bad_dir = bad_dir
        self.output_csv = output_csv
        
        # Ensures the data folder exists
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        self.pose_detector = PoseDetector(
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smoothing_window=5
        )
        self.feature_extractor = FeatureExtractor()
        self.global_timestamp_ms = 0  # Need this so MediaPipe timestamps strictly increase always

    def _process_video(self, video_path, label):
        """
        Extracts features from every frame of a specific video.
        
        Args:
            video_path (str): The .mp4 file.
            label (int): 1 for 'good', 0 for 'bad'.
            
        Returns:
            list: A list of dictionaries (rows for our CSV).
        """
        import cv2

        print(f"    -> Analyzing: {os.path.basename(video_path)} (Label: {'GOOD' if label==1 else 'BAD'})")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"       [WARNING] Could not open {video_path}")
            return []

        fps_orig = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_data = []
        frame_counter = 0
        valid_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_counter += 1
            
            # MediaPipe tasks API expects strictly increasing timestamps across its whole lifecycle.
            # Incrementing by a flat 33ms guarantees mathematically that it never goes backwards,
            # even if the next video has a drastically different framerate.
            self.global_timestamp_ms += 33
            
            # Step 1: Detect skeleton
            # We don't care about drawing the frame here, we just want the numbers.
            _, landmarks_norm = self.pose_detector.process_frame(frame, self.global_timestamp_ms)
            
            # Step 2: Extract mathematical features (angles)
            features = self.feature_extractor.extract_features(landmarks_norm)
            
            # Step 3: If we successfully extracted features, save them
            if features is not None:
                # Add metadata to the row
                features['video_name'] = os.path.basename(video_path)
                features['frame'] = frame_counter
                features['label'] = label
                
                video_data.append(features)
                valid_frames += 1

        cap.release()
        print(f"       Extracted {valid_frames} valid data frames out of {total_frames} total.")
        return video_data

    def build(self):
        """
        Main method to execute the batch processing.
        """
        print("\n" + "="*60)
        print("  CRICKET BIOMECHANICS — DATASET BUILDER (Sprint 2)")
        print("="*60)
        
        good_videos = glob.glob(os.path.join(self.good_dir, "*.mp4"))
        bad_videos = glob.glob(os.path.join(self.bad_dir, "*.mp4"))
        
        print(f"  Found {len(good_videos)} 'Good' videos.")
        print(f"  Found {len(bad_videos)} 'Bad' videos.")
        
        if len(good_videos) == 0 and len(bad_videos) == 0:
            print("  ❌ ERROR: No videos found to process!")
            return None

        all_dataset_rows = []
        start_time = time.time()

        # Process Good Videos
        print("\n  [ Phase 1/2 ] Processing GOOD Technique Videos...")
        for vid in good_videos:
            rows = self._process_video(vid, label=1)
            all_dataset_rows.extend(rows)
            
        # Process Bad Videos
        print("\n  [ Phase 2/2 ] Processing BAD Technique Videos...")
        for vid in bad_videos:
            rows = self._process_video(vid, label=0)
            all_dataset_rows.extend(rows)

        print("\n" + "="*60)
        if len(all_dataset_rows) == 0:
            print("  ❌ ERROR: Extraction failed. No usable frames found.")
            print("     This might mean the AI couldn't clearly see the players in the videos.")
            return None

        # Create Pandas DataFrame and save to CSV
        df = pd.DataFrame(all_dataset_rows)
        
        # Reorder columns so metadata is first, then features, then label last
        cols = ['video_name', 'frame', 'right_elbow_angle', 'left_elbow_angle', 
                'right_knee_bend', 'left_knee_bend', 'label']
        df = df[cols]
        
        df.to_csv(self.output_csv, index=False)
        
        elapsed = time.time() - start_time
        print(f"  ✅ DATASET GENERATION COMPLETE!")
        print(f"  Total Data Points (Frames): {len(df)}")
        print(f"  Saved CSV to              : {self.output_csv}")
        print(f"  Time Taken                : {elapsed:.1f} seconds")
        print("="*60 + "\n")
        
        self.pose_detector.close()
        return df

