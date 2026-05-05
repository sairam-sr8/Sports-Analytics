"""
========================================================
phase_detector.py — Temporal Intelligence Engine (V2)
========================================================

WHAT THIS FILE DOES:
--------------------
Detects WHICH PHASE of the batting motion the player is in,
frame by frame. This enables contextual, phase-aware scoring
instead of treating all frames identically.

CRICKET BATTING PHASES:
  0. SETUP    — Player is stationary in stance, not yet moving
  1. STANCE   — Weight loading, trigger movement begins
  2. BACKSWING — Bat rising, hips beginning to coil (backlift phase)
  3. DOWNSWING — Bat descending rapidly toward the ball
  4. IMPACT    — Bat-ball contact zone (peak power transfer)
  5. FOLLOW_THROUGH — Post-contact, bat continuing upward/across

DETECTION STRATEGY:
  We use a sliding window over wrist Y-position (in normalized frame coords).
  - Wrist moving UP   → BACKSWING
  - Wrist moving DOWN → DOWNSWING  
  - Wrist movement stops at low point → IMPACT
  - Wrist moving UP again after impact → FOLLOW_THROUGH
  - Minimal movement → SETUP/STANCE

We also cross-reference with elbow angle and hip rotation
for more accurate phase boundary detection.

AUTHOR: Cricket Biomechanics Analyzer V2
"""

from collections import deque
import numpy as np


# Phase constants
PHASE_SETUP          = 0
PHASE_STANCE         = 1
PHASE_BACKSWING      = 2
PHASE_DOWNSWING      = 3
PHASE_IMPACT         = 4
PHASE_FOLLOW_THROUGH = 5

PHASE_NAMES = {
    PHASE_SETUP:          "Setup",
    PHASE_STANCE:         "Stance",
    PHASE_BACKSWING:      "Backswing (Up)",
    PHASE_DOWNSWING:      "Downswing (Down)",
    PHASE_IMPACT:         "Impact",
    PHASE_FOLLOW_THROUGH: "Follow-Through",
}

PHASE_COLORS = {
    PHASE_SETUP:          (150, 150, 150),
    PHASE_STANCE:         (255, 200, 0),
    PHASE_BACKSWING:      (0, 200, 255),
    PHASE_DOWNSWING:      (0, 100, 255),
    PHASE_IMPACT:         (0, 255, 0),
    PHASE_FOLLOW_THROUGH: (0, 200, 150),
}


class PhaseDetector:
    """
    Classifies the current batting phase using temporal wrist kinematics.
    """

    def __init__(self, window=8, movement_threshold=0.003):
        """
        Args:
            window (int):               Frames to look back for velocity calc.
            movement_threshold (float): Minimum normalized displacement to 
                                        consider as "movement" (not noise).
        """
        self._wrist_y_history = deque(maxlen=window)
        self._wrist_x_history = deque(maxlen=window)
        self._elbow_history   = deque(maxlen=window)
        self._phase_history   = deque(maxlen=4)  # For smoothing phase transitions
        self._window          = window
        self._thresh          = movement_threshold

        # Track swing lifecycle state machine
        self._seen_backswing  = False
        self._seen_impact     = False
        self._frames_since_impact = 0

    def detect_phase(self, features: dict) -> tuple:
        """
        Determine the current batting phase from features.

        Args:
            features (dict): Output of FeatureExtractor.extract_features().

        Returns:
            tuple: (phase_id: int, phase_name: str, confidence: float 0-1)
        """
        if not features:
            return (PHASE_SETUP, PHASE_NAMES[PHASE_SETUP], 0.5)

        # Extract key kinematic signals
        # wrist Y: smaller value = higher on screen (inverted Y axis)
        wrist_y     = features.get('backlift_height', 0.0)   # relative to shoulder
        swing_arc   = features.get('bat_swing_arc', 0.0)
        elbow_angle = features.get('right_elbow_angle', 150.0)
        hip_rot     = features.get('hip_rotation_angle', 0.0)

        # Update history
        self._wrist_y_history.append(wrist_y)
        self._elbow_history.append(elbow_angle)

        # Need at least a few frames to make a sensible determination
        if len(self._wrist_y_history) < 3:
            return (PHASE_STANCE, PHASE_NAMES[PHASE_STANCE], 0.5)

        # ── Compute wrist velocity ────────────────────────────
        wrist_vals  = list(self._wrist_y_history)
        recent_vel  = wrist_vals[-1] - wrist_vals[-2]   # Instantaneous velocity
        window_vel  = wrist_vals[-1] - wrist_vals[0]    # Window-level trend

        # ── Phase Logic ───────────────────────────────────────
        # Phase is determined by the combination of:
        # 1. Direction of wrist movement
        # 2. Swing arc magnitude (speed)
        # 3. Elbow angle (extension → impact zone)
        # 4. Prior phase (state machine)

        # SETUP: Almost no movement at all (very tight gate — real swings have arc > 0.005)
        if swing_arc < self._thresh * 2 and abs(window_vel) < self._thresh:
            self._seen_backswing = False
            self._seen_impact    = False
            phase = PHASE_SETUP if hip_rot < 10 else PHASE_STANCE
            confidence = 0.7 if swing_arc < self._thresh else 0.5

        # BACKSWING: Wrist is moving upward (wrist Y relative to shoulder becomes more negative)
        elif window_vel < -self._thresh:
            self._seen_backswing = True
            phase = PHASE_BACKSWING
            confidence = min(1.0, abs(window_vel) / (self._thresh * 5))

        # DOWNSWING / IMPACT: Wrist moving downward (Y increasing)
        elif window_vel > self._thresh and self._seen_backswing:
            elbow_avg = np.mean(list(self._elbow_history))
            # Impact zone: elbow is most extended (higher angle) + active swing
            if elbow_avg > 130 and swing_arc > 0.04:
                phase = PHASE_IMPACT
                self._seen_impact = True
                self._frames_since_impact = 0
                confidence = min(1.0, swing_arc / 0.3)
            else:
                phase = PHASE_DOWNSWING
                confidence = min(1.0, abs(window_vel) / (self._thresh * 5))

        # FOLLOW THROUGH: After impact, wrist moving upward again
        elif self._seen_impact:
            self._frames_since_impact += 1
            phase = PHASE_FOLLOW_THROUGH
            confidence = min(1.0, self._frames_since_impact / 15.0)
            # Reset after a full follow-through is complete
            if self._frames_since_impact > 30:
                self._seen_backswing = False
                self._seen_impact    = False

        # Default: STANCE (loaded, waiting)
        else:
            phase = PHASE_STANCE
            confidence = 0.5

        # ── Smooth phase transitions ──────────────────────────
        self._phase_history.append(phase)
        # Majority vote from recent frames to avoid flicker
        if len(self._phase_history) >= 3:
            from collections import Counter
            most_common = Counter(self._phase_history).most_common(1)[0][0]
            phase = most_common

        return (phase, PHASE_NAMES[phase], round(float(confidence), 2))

    def reset(self):
        """Reset state for a new video session."""
        self._wrist_y_history.clear()
        self._wrist_x_history.clear()
        self._elbow_history.clear()
        self._phase_history.clear()
        self._seen_backswing  = False
        self._seen_impact     = False
        self._frames_since_impact = 0
