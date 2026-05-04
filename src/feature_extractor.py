"""
========================================================
feature_extractor.py — The Math Engine (V2 — 22 Features)
========================================================

V2 UPGRADE: Expanded from 4 → 22 features for enterprise-grade
sports biomechanics analysis, matching elite cricket analytics.

FEATURES WE EXTRACT PER FRAME:
---- ARM MECHANICS ----
  1.  right_elbow_angle         — Shoulder → Elbow → Wrist (bat arm)
  2.  left_elbow_angle          — Shoulder → Elbow → Wrist (guide arm)
  3.  right_shoulder_elevation  — Angle of right arm relative to torso
  4.  left_shoulder_elevation   — Angle of left arm relative to torso
  5.  shoulder_rotation_angle   — Rotational separation between shoulders (power source)
  6.  wrist_height_diff         — Vertical separation R-Wrist vs L-Wrist (backlift indicator)
  7.  bat_swing_arc             — Rate of change of wrist angle (swing velocity proxy)

---- LEG MECHANICS ----
  8.  right_knee_bend           — Hip → Knee → Ankle (back leg)
  9.  left_knee_bend            — Hip → Knee → Ankle (front leg, crucial for balance)
  10. right_hip_angle           — Shoulder → Hip → Knee (back leg drive)
  11. left_hip_angle            — Shoulder → Hip → Knee (front leg extension)

---- CORE MECHANICS ----
  12. hip_rotation_angle        — Separation angle between hip line and shoulder line
  13. trunk_lean_angle          — Tilt of torso from vertical (forward lean)
  14. spine_angle               — Midpoint of shoulders vs midpoint of hips angle

---- BALANCE & STABILITY ----
  15. center_of_mass_x          — Estimated CoM horizontal position (normalized 0-1)
  16. center_of_mass_y          — Estimated CoM vertical position (normalized 0-1)
  17. feet_width_ratio          — Ankle distance / shoulder width (stance width)
  18. weight_distribution       — L-ankle vs R-ankle horizontal vs CoM (balance check)

---- HEAD & FOLLOW-THROUGH ----
  19. head_stability_x          — Nose X deviation from shoulder midpoint (head movement)
  20. head_stability_y          — Nose Y deviation from shoulder midpoint (head drop)
  21. follow_through_angle      — Angle of arms at end of swing phase
  22. backlift_height           — R-Wrist Y position relative to shoulder (backlift height)

AUTHOR: Cricket Biomechanics Analyzer Project V2
"""

import numpy as np
from collections import deque


class FeatureExtractor:
    """
    Calculates 22 biomechanical metrics from MediaPipe landmarks.
    This transforms the project from "basic ML" to a genuine 
    sports biomechanics analysis system.
    """

    # ── MediaPipe Landmark Indices ─────────────────────────────
    NOSE           = 0
    L_EYE_INNER   = 1
    R_EYE_INNER   = 4
    L_SHOULDER     = 11
    R_SHOULDER     = 12
    L_ELBOW        = 13
    R_ELBOW        = 14
    L_WRIST        = 15
    R_WRIST        = 16
    L_HIP          = 23
    R_HIP          = 24
    L_KNEE         = 25
    R_KNEE         = 26
    L_ANKLE        = 27
    R_ANKLE        = 28
    L_HEEL         = 29
    R_HEEL         = 30
    L_FOOT_INDEX   = 31
    R_FOOT_INDEX   = 32

    def __init__(self, history_window=5):
        """
        Args:
            history_window (int): Frames of history for temporal features
                                  like bat_swing_arc.
        """
        # We keep a short history of wrist angles to compute swing arc rate
        self._wrist_history = deque(maxlen=history_window)

    # ────────────────────────────────────────────────────────────
    #  CORE MATH HELPERS
    # ────────────────────────────────────────────────────────────

    def _calculate_angle(self, a, b, c):
        """
        Calculates the angle at vertex B formed by rays BA and BC.

        Args:
            a, b, c: (x, y) tuples.
        Returns:
            float: Angle in degrees [0, 180].
        """
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        c = np.array(c, dtype=np.float64)

        ba = a - b
        bc = c - b

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-9 or norm_bc < 1e-9:
            return 0.0

        cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine = np.clip(cosine, -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cosine))), 2)

    def _vec_angle_2d(self, v1, v2):
        """
        Signed angle between two 2D vectors (in degrees).
        Positive = counter-clockwise.
        """
        v1 = np.array(v1, dtype=np.float64)
        v2 = np.array(v2, dtype=np.float64)
        angle = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
        return round(float((angle + 180) % 360 - 180), 2)

    def _get_xy(self, landmarks, idx):
        """Returns (x, y) tuple for a landmark, or None if not visible."""
        if idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        if lm.visibility < 0.4:
            return None
        return (lm.x, lm.y)

    def _get_xyz(self, landmarks, idx):
        """Returns (x, y, z) for a landmark regardless of visibility."""
        if idx >= len(landmarks):
            return (0.5, 0.5, 0.0)
        lm = landmarks[idx]
        return (lm.x, lm.y, lm.z)

    def _midpoint(self, p1, p2):
        """Midpoint of two (x, y) tuples."""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    # ────────────────────────────────────────────────────────────
    #  MAIN EXTRACTION METHOD
    # ────────────────────────────────────────────────────────────

    def extract_features(self, landmarks):
        """
        Calculates all 22 batting biomechanics features for one frame.

        Args:
            landmarks: Smoothed landmarks list from PoseDetector.
                       (List of 33 objects with .x, .y, .z, .visibility)
        Returns:
            dict: 22-feature dictionary, or None if key joints not visible.
        """
        if landmarks is None:
            return None

        # ── GATE CHECK: We MUST see the right-side body (batting side) ──
        required_joints = [
            self.R_SHOULDER, self.R_ELBOW, self.R_WRIST,
            self.R_HIP,      self.R_KNEE,  self.R_ANKLE
        ]
        for j in required_joints:
            if j >= len(landmarks) or landmarks[j].visibility < 0.45:
                return None

        features = {}

        # ── CONVENIENCE COORDINATES ──────────────────────────────
        r_sh = (landmarks[self.R_SHOULDER].x, landmarks[self.R_SHOULDER].y)
        l_sh = (landmarks[self.L_SHOULDER].x, landmarks[self.L_SHOULDER].y)
        r_el = (landmarks[self.R_ELBOW].x,    landmarks[self.R_ELBOW].y)
        l_el = (landmarks[self.L_ELBOW].x,    landmarks[self.L_ELBOW].y)
        r_wr = (landmarks[self.R_WRIST].x,    landmarks[self.R_WRIST].y)
        l_wr = (landmarks[self.L_WRIST].x,    landmarks[self.L_WRIST].y)
        r_hp = (landmarks[self.R_HIP].x,      landmarks[self.R_HIP].y)
        l_hp = (landmarks[self.L_HIP].x,      landmarks[self.L_HIP].y)
        r_kn = (landmarks[self.R_KNEE].x,     landmarks[self.R_KNEE].y)
        l_kn = (landmarks[self.L_KNEE].x,     landmarks[self.L_KNEE].y)
        r_an = (landmarks[self.R_ANKLE].x,    landmarks[self.R_ANKLE].y)
        l_an = (landmarks[self.L_ANKLE].x,    landmarks[self.L_ANKLE].y)
        nose  = (landmarks[self.NOSE].x,       landmarks[self.NOSE].y)

        sh_mid = self._midpoint(l_sh, r_sh)
        hp_mid = self._midpoint(l_hp, r_hp)

        # ╔══════════════════════════════════════════════════════╗
        # ║   FEATURE GROUP 1: ARM MECHANICS (7 features)       ║
        # ╚══════════════════════════════════════════════════════╝

        # 1. Right Elbow Angle (primary batting arm)
        features['right_elbow_angle'] = self._calculate_angle(r_sh, r_el, r_wr)

        # 2. Left Elbow Angle (guide arm)
        features['left_elbow_angle'] = 0.0
        if landmarks[self.L_ELBOW].visibility > 0.4 and landmarks[self.L_WRIST].visibility > 0.4:
            features['left_elbow_angle'] = self._calculate_angle(l_sh, l_el, l_wr)

        # 3. Right Shoulder Elevation — angle of upper arm to torso vertical
        features['right_shoulder_elevation'] = self._calculate_angle(r_hp, r_sh, r_el)

        # 4. Left Shoulder Elevation
        features['left_shoulder_elevation'] = 0.0
        if landmarks[self.L_SHOULDER].visibility > 0.4:
            features['left_shoulder_elevation'] = self._calculate_angle(l_hp, l_sh, l_el)

        # 5. Shoulder Rotation Angle — how much shoulders have rotated
        # We project the shoulder line onto the horizontal axis.
        # 0° = fully open (facing camera), 90° = fully sideways (ideal batting stance)
        shoulder_vec = (r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
        hip_vec      = (r_hp[0] - l_hp[0], r_hp[1] - l_hp[1])
        features['shoulder_rotation_angle'] = abs(self._vec_angle_2d(shoulder_vec, (1, 0)))

        # 6. Wrist Height Difference — backlift/follow-through proxy
        # Positive = R-wrist higher than L-wrist (good backlift)
        # Y is inverted in image coords: lower y = higher on screen
        features['wrist_height_diff'] = round(float(l_wr[1] - r_wr[1]), 4)

        # 7. Bat Swing Arc — rate of change of r_wrist position (velocity proxy)
        current_wrist_pos = np.array(r_wr)
        self._wrist_history.append(current_wrist_pos)
        if len(self._wrist_history) >= 2:
            wrist_displacement = np.linalg.norm(
                self._wrist_history[-1] - self._wrist_history[0]
            )
        else:
            wrist_displacement = 0.0
        features['bat_swing_arc'] = round(float(wrist_displacement), 4)

        # ╔══════════════════════════════════════════════════════╗
        # ║   FEATURE GROUP 2: LEG MECHANICS (4 features)       ║
        # ╚══════════════════════════════════════════════════════╝

        # 8. Right Knee Bend (back leg — power coil)
        features['right_knee_bend'] = self._calculate_angle(r_hp, r_kn, r_an)

        # 9. Left Knee Bend (front leg — balance pillar)
        features['left_knee_bend'] = 0.0
        if (landmarks[self.L_HIP].visibility > 0.4 and
            landmarks[self.L_KNEE].visibility > 0.4 and
            landmarks[self.L_ANKLE].visibility > 0.4):
            features['left_knee_bend'] = self._calculate_angle(l_hp, l_kn, l_an)

        # 10. Right Hip Angle (torso-to-back-leg linkage)
        features['right_hip_angle'] = self._calculate_angle(r_sh, r_hp, r_kn)

        # 11. Left Hip Angle (torso-to-front-leg linkage)
        features['left_hip_angle'] = 0.0
        if landmarks[self.L_HIP].visibility > 0.4:
            features['left_hip_angle'] = self._calculate_angle(l_sh, l_hp, l_kn)

        # ╔══════════════════════════════════════════════════════╗
        # ║   FEATURE GROUP 3: CORE MECHANICS (3 features)      ║
        # ╚══════════════════════════════════════════════════════╝

        # 12. Hip Rotation Angle — how much hips lead the shoulders
        features['hip_rotation_angle'] = abs(self._vec_angle_2d(hip_vec, (1, 0)))

        # 13. Trunk Lean Angle — forward lean of spine from vertical
        # Spine = midpoint(shoulders) → midpoint(hips)
        spine_vec = (sh_mid[0] - hp_mid[0], sh_mid[1] - hp_mid[1])
        # Vertical vector pointing up (inverted Y in image)
        features['trunk_lean_angle'] = abs(self._vec_angle_2d(spine_vec, (0, -1)))

        # 14. Spine Angle — angle at midpoint shoulder using hips and head
        features['spine_angle'] = self._calculate_angle(hp_mid, sh_mid, nose)

        # ╔══════════════════════════════════════════════════════╗
        # ║   FEATURE GROUP 4: BALANCE & STABILITY (4 features) ║
        # ╚══════════════════════════════════════════════════════╝

        # 15 & 16. Center of Mass (estimated from torso landmarks)
        # Weighted average of key body segments
        com_x = (r_sh[0] + l_sh[0] + r_hp[0] + l_hp[0]) / 4.0
        com_y = (r_sh[1] + l_sh[1] + r_hp[1] + l_hp[1]) / 4.0
        features['center_of_mass_x'] = round(float(com_x), 4)
        features['center_of_mass_y'] = round(float(com_y), 4)

        # 17. Feet Width Ratio — ankle distance / shoulder width
        ankle_width    = abs(r_an[0] - l_an[0])
        shoulder_width = abs(r_sh[0] - l_sh[0])
        feet_width_ratio = (ankle_width / shoulder_width) if shoulder_width > 0.01 else 1.0
        features['feet_width_ratio'] = round(float(feet_width_ratio), 4)

        # 18. Weight Distribution — how evenly balanced between feet
        # 0 = fully on right foot, 1 = fully on left foot, 0.5 = perfectly balanced
        if (landmarks[self.L_ANKLE].visibility > 0.4 and
            landmarks[self.R_ANKLE].visibility > 0.4):
            ankle_mid_x = (l_an[0] + r_an[0]) / 2.0
            foot_spread = abs(l_an[0] - r_an[0])
            if foot_spread > 0.01:
                weight_dist = (com_x - r_an[0]) / foot_spread
            else:
                weight_dist = 0.5
        else:
            weight_dist = 0.5
        features['weight_distribution'] = round(float(np.clip(weight_dist, 0.0, 1.0)), 4)

        # ╔══════════════════════════════════════════════════════╗
        # ║   FEATURE GROUP 5: HEAD & SWING (4 features)        ║
        # ╚══════════════════════════════════════════════════════╝

        # 19. Head Stability X — how much the head is drifting horizontally
        # Good technique: head stays still over the ball (close to shoulder midpoint)
        features['head_stability_x'] = round(float(nose[0] - sh_mid[0]), 4)

        # 20. Head Stability Y — vertical head drop during swing
        features['head_stability_y'] = round(float(nose[1] - sh_mid[1]), 4)

        # 21. Follow-Through Angle — angle of full arm extension at end
        # Same as right elbow angle but measured from hip perspective for full arc
        features['follow_through_angle'] = self._calculate_angle(r_hp, r_sh, r_wr)

        # 22. Backlift Height — R-Wrist Y relative to R-Shoulder Y
        # Negative means wrist is ABOVE shoulder (great backlift), positive = below
        features['backlift_height'] = round(float(r_wr[1] - r_sh[1]), 4)

        return features


    def get_feature_names(self):
        """Returns ordered list of all 22 feature names."""
        return [
            'right_elbow_angle',       'left_elbow_angle',
            'right_shoulder_elevation','left_shoulder_elevation',
            'shoulder_rotation_angle', 'wrist_height_diff',
            'bat_swing_arc',
            'right_knee_bend',         'left_knee_bend',
            'right_hip_angle',         'left_hip_angle',
            'hip_rotation_angle',      'trunk_lean_angle',
            'spine_angle',
            'center_of_mass_x',        'center_of_mass_y',
            'feet_width_ratio',        'weight_distribution',
            'head_stability_x',        'head_stability_y',
            'follow_through_angle',    'backlift_height',
        ]
