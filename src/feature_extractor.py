"""
========================================================
feature_extractor.py — The Math Engine
========================================================

WHAT THIS FILE DOES (In Plain English):
----------------------------------------
It takes the raw X, Y coordinates from our PoseDetector
and uses trigonometry to calculate actual human angles.

For example, to find the "Elbow Angle":
We take the Shoulder (A), Elbow (B), and Wrist (C).
We calculate the angle formed at B.
- If angle is ~180 degrees -> arm is perfectly straight
- If angle is ~90 degrees  -> arm is bent in an L-shape

FEATURES WE EXTRACT PER FRAME:
1. right_elbow_angle
2. left_elbow_angle
3. right_knee_bend
4. left_knee_bend
...and we can add more later (like shoulder alignment).

AUTHOR: Cricket Biomechanics Analyzer Project
SPRINT: 2 — Feature Extraction & Dataset Generation
"""

import numpy as np

class FeatureExtractor:
    """
    Calculates biomechanical metrics (angles) from MediaPipe landmarks.
    """

    def __init__(self):
        # We need the indices of the landmarks to do the math.
        # These are the same MediaPipe numbers we used in PoseDetector.
        self.L_SHOULDER = 11
        self.R_SHOULDER = 12
        self.L_ELBOW = 13
        self.R_ELBOW = 14
        self.L_WRIST = 15
        self.R_WRIST = 16
        self.L_HIP = 23
        self.R_HIP = 24
        self.L_KNEE = 25
        self.R_KNEE = 26
        self.L_ANKLE = 27
        self.R_ANKLE = 28

    def _calculate_angle(self, a, b, c):
        """
        Calculates the angle at point 'b' between lines ab and bc.
        
        Args:
            a, b, c: Tuples of (x, y) coordinates. Example: (0.5, 0.3)
            
        Returns:
            float: The angle in degrees (0 to 180).
        """
        a = np.array(a) # First point
        b = np.array(b) # Middle point (the vertex)
        c = np.array(c) # End point
        
        # Calculate the vectors pointing away from the vertex (b)
        ba = a - b
        bc = c - b
        
        # Trigonometry: dot product and magnitude to find the cosine of the angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Prevent floating point errors from going slightly outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        # Convert from radians to degrees
        angle = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle)
        
        return round(angle_degrees, 2)

    def extract_features(self, landmarks):
        """
        Calculates all our crucial batting angles for a single frame.

        Args:
            landmarks: The smoothed landmarks list from our PoseDetector.
                       (List of 33 objects with .x, .y, .visibility)

        Returns:
            dict: A dictionary of calculated angles. Returns None if 
                  key landmarks are not visible.
        """
        if landmarks is None:
            return None

        # Check visibility. If we can't see the wrist, we can't calculate elbow angle.
        # We require at least 50% visibility for key joints.
        required_joints = [self.R_SHOULDER, self.R_ELBOW, self.R_WRIST, 
                           self.R_HIP, self.R_KNEE, self.R_ANKLE]
        
        for joint in required_joints:
            # Check if landmark is within bounds and visible
            if joint >= len(landmarks) or landmarks[joint].visibility < 0.5:
                # If we cannot clearly see the right arm and leg, we skip this frame's math.
                # In cricket data, bad frames happen (ball block, bat block, motion blur).
                return None

        features = {}

        # 1. Right Elbow Angle (Shoulder -> Elbow -> Wrist)
        # --------------------------------------------------
        r_shoulder = (landmarks[self.R_SHOULDER].x, landmarks[self.R_SHOULDER].y)
        r_elbow    = (landmarks[self.R_ELBOW].x, landmarks[self.R_ELBOW].y)
        r_wrist    = (landmarks[self.R_WRIST].x, landmarks[self.R_WRIST].y)
        
        features['right_elbow_angle'] = self._calculate_angle(r_shoulder, r_elbow, r_wrist)

        # 2. Right Knee Bend (Hip -> Knee -> Ankle)
        # --------------------------------------------------
        r_hip   = (landmarks[self.R_HIP].x, landmarks[self.R_HIP].y)
        r_knee  = (landmarks[self.R_KNEE].x, landmarks[self.R_KNEE].y)
        r_ankle = (landmarks[self.R_ANKLE].x, landmarks[self.R_ANKLE].y)
        
        features['right_knee_bend'] = self._calculate_angle(r_hip, r_knee, r_ankle)


        # Optional: Let's grab the left side too, if visible
        # We won't strictly enforce left side visibility, we'll just put 0.0 if missing.
        features['left_elbow_angle'] = 0.0
        features['left_knee_bend'] = 0.0
        
        if (landmarks[self.L_SHOULDER].visibility > 0.5 and 
            landmarks[self.L_ELBOW].visibility > 0.5 and 
            landmarks[self.L_WRIST].visibility > 0.5):
            
            l_sh = (landmarks[self.L_SHOULDER].x, landmarks[self.L_SHOULDER].y)
            l_el = (landmarks[self.L_ELBOW].x, landmarks[self.L_ELBOW].y)
            l_wr = (landmarks[self.L_WRIST].x, landmarks[self.L_WRIST].y)
            features['left_elbow_angle'] = self._calculate_angle(l_sh, l_el, l_wr)

        if (landmarks[self.L_HIP].visibility > 0.5 and 
            landmarks[self.L_KNEE].visibility > 0.5 and 
            landmarks[self.L_ANKLE].visibility > 0.5):
            
            l_hp = (landmarks[self.L_HIP].x, landmarks[self.L_HIP].y)
            l_kn = (landmarks[self.L_KNEE].x, landmarks[self.L_KNEE].y)
            l_an = (landmarks[self.L_ANKLE].x, landmarks[self.L_ANKLE].y)
            features['left_knee_bend'] = self._calculate_angle(l_hp, l_kn, l_an)

        return features
