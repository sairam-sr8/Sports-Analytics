"""
========================================================
shot_classifier.py — Shot Type AI (V2)
========================================================

WHAT THIS FILE DOES:
--------------------
Classifies the type of cricket shot being played using a
sliding window of biomechanical features. Shot type fundamentally
changes which biomechanical standards should be applied.

SHOT TYPES DETECTED:
  0. UNKNOWN      — Cannot determine yet (insufficient data)
  1. COVER_DRIVE  — Off-side, expansive, ball in front of stumps
  2. PULL_SHOT    — Short ball, horizontal bat, arms above shoulder
  3. STRAIGHT_DRIVE — Straight down the ground, vertical bat
  4. SWEEP_SHOT   — Low trajectory, knee down, bat sweeps across
  5. DEFENSIVE    — Compact, minimal foot movement, soft hands

CLASSIFICATION METHOD:
  Rule-based heuristics from biomechanics thresholds.
  (Can be upgraded to LSTM later with labelled data.)
  
KEY DISCRIMINATORS:
  - shoulder_rotation_angle → High = expansive shot (cover drive)
  - trunk_lean_angle        → Low = square-on shot (pull), High = drive
  - wrist_height_diff       → Positive high = arms above (pull shot)
  - right_elbow_angle       → Extended = drive, Bent = defensive
  - right_shoulder_elevation → High = pull, Low = sweep

AUTHOR: Cricket Biomechanics Analyzer V2
"""

from collections import deque, Counter
import numpy as np
import joblib
import os
import pandas as pd
import sys

# Ensure we can import MLTrainer to get FEATURE_COLUMNS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_trainer import MLTrainer

# Shot ID constants
SHOT_UNKNOWN       = 0
SHOT_COVER_DRIVE   = 1
SHOT_PULL_SHOT     = 2
SHOT_STRAIGHT_DRIVE = 3
SHOT_SWEEP_SHOT    = 4
SHOT_DEFENSIVE     = 5

SHOT_NAMES = {
    SHOT_UNKNOWN:        "Analyzing...",
    SHOT_COVER_DRIVE:   "Cover Drive",
    SHOT_PULL_SHOT:     "Pull Shot",
    SHOT_STRAIGHT_DRIVE: "Straight Drive",
    SHOT_SWEEP_SHOT:    "Sweep Shot",
    SHOT_DEFENSIVE:     "Defensive Block",
}

SHOT_EMOJIS = {
    SHOT_UNKNOWN:        "🔍",
    SHOT_COVER_DRIVE:   "🏏➡️",
    SHOT_PULL_SHOT:     "💪⬆️",
    SHOT_STRAIGHT_DRIVE: "🏏⬆️",
    SHOT_SWEEP_SHOT:    "🏏⬇️",
    SHOT_DEFENSIVE:     "🛡️",
}

SHOT_DESCRIPTIONS = {
    SHOT_UNKNOWN:        "Collecting data...",
    SHOT_COVER_DRIVE:   "Off-side shot through cover region",
    SHOT_PULL_SHOT:     "Attacking short ball over square leg",
    SHOT_STRAIGHT_DRIVE: "Driving straight down the ground",
    SHOT_SWEEP_SHOT:    "Sweeping across the line of the ball",
    SHOT_DEFENSIVE:     "Compact block, soft hands",
}

class ShotClassifier:
    """
    Classifies shot type from a sliding window of feature frames.

    CALIBRATED to real dataset feature ranges (measured from actual cricket videos):
      Cover Drive  : shoulder_rot ~124°, hip_rot ~136°, elbow ~145°, swing_arc ~0.092
      Straight Drive: shoulder_rot ~41°,  hip_rot ~49°,  elbow ~151°, swing_arc ~0.062
      Sweep Shot   : left_knee ~90°,     shoulder_rot ~144°, hip_rot ~147°
      Pull Shot    : wrist_diff < -0.05, swing_arc ~0.161, shoulder_elev ~69°
      Defensive    : elbow ~60°,          swing_arc ~0.037, hip_rot ~27°

    NOTE: ML model is disabled — it was trained on mislabeled data.
          Only calibrated heuristics are used.
    """

    # Minimum swing arc to consider a shot is happening (stationary stance threshold)
    _MIN_SWING = 0.015

    def __init__(self, window=20, model_path="models/shot_classifier.pkl"):
        self._history = deque(maxlen=window)
        self._shot_history = deque(maxlen=7)
        # ML model intentionally disabled — trained on mislabeled data
        self.model = None
        self.model_loaded = False
        self.feature_columns = []

    def classify_frame(self, features: dict) -> tuple:
        """
        Update the feature window and return the current shot classification.
        Returns: (shot_id: int, shot_name: str, confidence: float)
        """
        if not features:
            return (SHOT_UNKNOWN, SHOT_NAMES[SHOT_UNKNOWN], 0.0)

        self._history.append(features)

        # Need at least 5 frames to make a stable classification
        if len(self._history) < 5:
            return (SHOT_UNKNOWN, SHOT_NAMES[SHOT_UNKNOWN], 0.2)

        # Classify using PEAK frames (top 30% by swing arc) not all frames equally
        # This ensures we classify the actual swing, not the setup stance
        shot_id, confidence = self._classify_peak_frames()

        # Smooth output with recent shot history (majority vote, 7 frames)
        self._shot_history.append(shot_id)
        if len(self._shot_history) >= 4:
            most_common = Counter(self._shot_history).most_common(1)[0][0]
            shot_id = most_common

        return (shot_id, SHOT_NAMES[shot_id], round(float(confidence), 2))

    def _classify_peak_frames(self) -> tuple:
        """
        Average the top 30% of frames (by bat_swing_arc) and classify from those.
        This isolates the actual swing motion from the setup/stance frames.
        """
        frames = list(self._history)

        # Sort by swing arc descending, take top 30% (minimum 3 frames)
        sorted_frames = sorted(frames, key=lambda f: f.get('bat_swing_arc', 0.0), reverse=True)
        top_n = max(3, len(sorted_frames) // 3)
        peak_frames = sorted_frames[:top_n]

        # Check if there is meaningful swing motion in the peak frames
        peak_swing = np.mean([f.get('bat_swing_arc', 0.0) for f in peak_frames])
        if peak_swing < self._MIN_SWING:
            return (SHOT_UNKNOWN, 0.3)

        # Average features across peak frames
        keys = frames[0].keys()
        avg = {k: float(np.mean([f.get(k, 0.0) for f in peak_frames])) for k in keys}

        return self._classify_from_averages(avg)

    def _average_features(self) -> dict:
        """Compute mean of all features in the sliding window."""
        if not self._history:
            return {}
        keys = self._history[0].keys()
        return {
            k: float(np.mean([f.get(k, 0.0) for f in self._history]))
            for k in keys
        }

    def _classify_from_averages(self, avg: dict) -> tuple:
        """
        Apply CALIBRATED biomechanical decision rules to classify shot type.
        Thresholds are derived from real feature range analysis of the training dataset.

        Key discriminators (real measured values):
          shoulder_rotation_angle:
            - Cover Drive  : ~124°  (high — turning into off side)
            - Straight Drive: ~41°  (low  — facing straight)
            - Sweep Shot   : ~144°  (very high)
            - Pull Shot    : ~88°   (moderate)
            - Defensive    : ~35°   (low)
          hip_rotation_angle:
            - Cover Drive  : ~136°  (high body rotation)
            - Straight Drive: ~49°  (low — body facing straight)
            - Sweep Shot   : ~147°  (very high)
          left_knee_bend:
            - Sweep Shot   : ~90°   (knee nearly touching ground — strongest indicator)
            - All others   : >107°
          right_elbow_angle:
            - Defensive    : ~60°   (very bent, soft hands)
            - All drives   : >135°  (extended arm)
          wrist_height_diff:
            - Pull Shot    : ~-0.085 (both wrists high, below shoulder level)
        """
        sh_rot     = avg.get('shoulder_rotation_angle', 90.0)
        hip_rot    = avg.get('hip_rotation_angle', 90.0)
        trunk_lean = avg.get('trunk_lean_angle', 15.0)
        wrist_diff = avg.get('wrist_height_diff', 0.0)
        r_elbow    = avg.get('right_elbow_angle', 150.0)
        r_sh_elev  = avg.get('right_shoulder_elevation', 50.0)
        swing_arc  = avg.get('bat_swing_arc', 0.0)
        l_knee     = avg.get('left_knee_bend', 130.0)
        r_knee     = avg.get('right_knee_bend', 150.0)

        # ── SWEEP SHOT ─────────────────────────────────────────────────
        # STRONGEST indicator: front knee bent deeply (kneeling position).
        # Real data: left_knee mean=90°. Threshold: < 115° is very safe.
        if l_knee < 115 and sh_rot > 80:
            confidence = min(1.0, 0.6 + (115 - l_knee) / 100)
            return (SHOT_SWEEP_SHOT, confidence)

        # ── PULL SHOT ──────────────────────────────────────────────────
        # Wrists are both high (above or near shoulder) + big swing arc.
        # Real data: wrist_diff mean=-0.085 (wrist well above shoulder).
        # Also: right_shoulder_elevation ~69° (arm raised high).
        if wrist_diff < -0.04 and r_sh_elev > 55 and swing_arc > 0.05:
            confidence = min(1.0, 0.5 + abs(wrist_diff) * 5)
            return (SHOT_PULL_SHOT, confidence)

        # ── DEFENSIVE BLOCK ────────────────────────────────────────────
        # Very bent elbow (~60°) is the clearest indicator — soft hands.
        # Real data: right_elbow mean=60°, hip_rot mean=27°.
        if r_elbow < 90 and hip_rot < 50:
            confidence = min(1.0, 0.5 + (90 - r_elbow) / 90)
            return (SHOT_DEFENSIVE, confidence)

        # ── COVER DRIVE vs STRAIGHT DRIVE ─────────────────────────────
        # The PRIMARY discriminator is shoulder_rotation_angle:
        # Cover Drive  : high rotation ~124°, hip_rot ~136°
        # Straight Drive: low rotation ~41°, hip_rot ~49°
        # Boundary: ~80-90° separates the two clearly.
        if r_elbow > 120 and swing_arc > 0.015:
            if sh_rot > 80 and hip_rot > 80:
                # High body rotation = off-side drive (Cover Drive)
                confidence = min(1.0, 0.5 + (sh_rot - 80) / 100)
                return (SHOT_COVER_DRIVE, confidence)
            elif sh_rot <= 80 and hip_rot <= 80:
                # Low body rotation = straight/on-side drive
                confidence = min(1.0, 0.5 + (80 - sh_rot) / 100)
                return (SHOT_STRAIGHT_DRIVE, confidence)
            else:
                # Mixed signals — lean toward Cover Drive if arm is high
                if sh_rot > 60:
                    return (SHOT_COVER_DRIVE, 0.55)
                else:
                    return (SHOT_STRAIGHT_DRIVE, 0.55)

        # Default — not enough motion or unclear posture
        return (SHOT_UNKNOWN, 0.3)

    def classify_video_features(self, all_features: list) -> tuple:
        """
        Classify the shot for an ENTIRE video by looking at the peak swing frames.
        This is used for the final session report — gives ONE definitive answer.

        Args:
            all_features: List of feature dicts from every processed frame.
        Returns:
            (shot_id, shot_name, confidence)
        """
        if not all_features:
            return (SHOT_UNKNOWN, SHOT_NAMES[SHOT_UNKNOWN], 0.0)

        # Sort all frames by swing arc — take top 20% (the real swing frames)
        sorted_frames = sorted(all_features, key=lambda f: f.get('bat_swing_arc', 0.0), reverse=True)
        top_n = max(5, len(sorted_frames) // 5)
        peak_frames = sorted_frames[:top_n]

        peak_swing = np.mean([f.get('bat_swing_arc', 0.0) for f in peak_frames])
        if peak_swing < self._MIN_SWING:
            return (SHOT_UNKNOWN, "No clear swing detected", 0.3)

        keys = peak_frames[0].keys()
        avg = {k: float(np.mean([f.get(k, 0.0) for f in peak_frames])) for k in keys}

        shot_id, confidence = self._classify_from_averages(avg)
        return (shot_id, SHOT_NAMES[shot_id], round(float(confidence), 2))

    def reset(self):
        """Reset classifier for a new video."""
        self._history.clear()
        self._shot_history.clear()
