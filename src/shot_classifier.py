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
    Uses biomechanical heuristics — no training data required.
    """

    def __init__(self, window=15, model_path="models/shot_classifier.pkl"):
        """
        Args:
            window (int): Number of frames to aggregate for classification.
                          Larger = more stable but more latency.
        """
        self._history = deque(maxlen=window)
        self._shot_history = deque(maxlen=5)
        self.model = None
        self.model_loaded = False
        
        # Try loading the ML model
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self.model_loaded = True
        except Exception as e:
            pass
            
        self.feature_columns = MLTrainer.FEATURE_COLUMNS

    def classify_frame(self, features: dict) -> tuple:
        """
        Update the feature window and return the current shot classification.

        Args:
            features (dict): Features dict from FeatureExtractor.

        Returns:
            tuple: (shot_id: int, shot_name: str, confidence: float)
        """
        if not features:
            return (SHOT_UNKNOWN, SHOT_NAMES[SHOT_UNKNOWN], 0.0)

        self._history.append(features)

        if len(self._history) < 5:
            return (SHOT_UNKNOWN, SHOT_NAMES[SHOT_UNKNOWN], 0.2)

        # Average key features over the window for stability
        avg = self._average_features()
        swing_arc = avg.get('bat_swing_arc', 0.0)

        # ML CLASSIFICATION
        # Only use ML if there is actual swing movement.
        # Threshold lowered to 0.008 to match real normalized coordinate values
        # (bat_swing_arc is typically 0.005-0.08 for actual swings).
        if self.model_loaded and self.model is not None and swing_arc >= 0.008:
            # Prepare dataframe row
            available_features = [f for f in self.feature_columns if f in avg]
            if len(available_features) > 10:  # Need minimum features
                # Use the ML model
                X = pd.DataFrame([avg])[available_features].fillna(0.0)
                try:
                    pred_name = self.model.predict(X)[0]
                    # Map the string name back to ID
                    shot_id = SHOT_UNKNOWN
                    for sid, sname in SHOT_NAMES.items():
                        if sname == pred_name:
                            shot_id = sid
                            break
                            
                    # Get prediction probability if possible
                    confidence = 0.8
                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(X)[0]
                        confidence = float(np.max(proba))
                        
                    # Still fallback to heuristic if ML is very uncertain
                    if confidence < 0.4:
                        shot_id, confidence = self._classify_from_averages(avg)
                        
                except Exception:
                    # Fallback to heuristic
                    shot_id, confidence = self._classify_from_averages(avg)
            else:
                shot_id, confidence = self._classify_from_averages(avg)
        else:
            # HEURISTIC CLASSIFICATION (Original behavior or if no swing)
            shot_id, confidence = self._classify_from_averages(avg)

        # Smooth with recent history
        self._shot_history.append(shot_id)
        if len(self._shot_history) >= 3:
            most_common = Counter(self._shot_history).most_common(1)[0][0]
            shot_id = most_common

        return (shot_id, SHOT_NAMES[shot_id], round(float(confidence), 2))

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
        Apply biomechanical decision rules to classify shot type.

        Returns:
            tuple: (shot_id, confidence)
        """
        sh_rot    = avg.get('shoulder_rotation_angle', 0.0)
        trunk_lean = avg.get('trunk_lean_angle', 15.0)
        wrist_diff = avg.get('wrist_height_diff', 0.0)
        r_elbow   = avg.get('right_elbow_angle', 150.0)
        r_sh_elev = avg.get('right_shoulder_elevation', 90.0)
        swing_arc = avg.get('bat_swing_arc', 0.0)
        r_knee    = avg.get('right_knee_bend', 155.0)
        hip_rot   = avg.get('hip_rotation_angle', 30.0)

        # ── SWEEP SHOT ────────────────────────────────────────
        # Very low back knee, bat going cross-bat, high shoulder rotation
        if r_knee < 110 and sh_rot > 60:
            return (SHOT_SWEEP_SHOT, 0.80)

        # ── PULL SHOT ─────────────────────────────────────────
        # Arms above shoulder level, cross-bat plane, short ball
        if wrist_diff < -0.10 and r_sh_elev > 100 and swing_arc > 0.08:
            confidence = min(1.0, 0.5 + (r_sh_elev - 100) / 80)
            return (SHOT_PULL_SHOT, confidence)

        # ── DEFENSIVE BLOCK ───────────────────────────────────
        # Very little swing arc, elbow bent, minimal hip rotation
        if swing_arc < 0.008 and r_elbow < 120 and hip_rot < 15:
            return (SHOT_DEFENSIVE, 0.85)

        # ── COVER DRIVE ───────────────────────────────────────
        # High shoulder rotation, good lean forward, arm extended, active swing
        if sh_rot > 40 and trunk_lean > 10 and r_elbow > 130 and hip_rot > 25 and swing_arc > 0.008:
            confidence = min(1.0, 0.5 + (sh_rot - 40) / 60)
            return (SHOT_COVER_DRIVE, confidence)

        # ── STRAIGHT DRIVE ────────────────────────────────────
        # Lower shoulder rotation (hitting straight), arm extended, active swing
        if sh_rot < 40 and r_elbow > 130 and trunk_lean > 8 and swing_arc > 0.008:
            confidence = min(1.0, 0.5 + r_elbow / 200)
            return (SHOT_STRAIGHT_DRIVE, confidence)

        # Default
        return (SHOT_UNKNOWN, 0.3)

    def reset(self):
        """Reset classifier for a new video."""
        self._history.clear()
        self._shot_history.clear()
