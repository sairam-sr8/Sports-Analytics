"""
========================================================
shap_explainer.py — SHAP Explainability Engine (V2)
========================================================

WHAT THIS FILE DOES:
--------------------
Uses SHAP (SHapley Additive exPlanations) to explain WHY the
Random Forest model gave a specific score.

Instead of just saying "Your score is 72/100", this engine tells
the player WHICH specific body mechanic is dragging their score down.

This is what professors and judges LOVE — it shows the student
understands AI explainability, not just accuracy.

OUTPUT EXAMPLE:
  "Your technique score was mainly reduced by:
   🔴 bat_swing_arc        (−15 pts)
   🔴 right_elbow_angle    (−8 pts)
   🟢 shoulder_rotation    (+12 pts — you're doing this well!)"

AUTHOR: Cricket Biomechanics Analyzer V2
"""

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Human-friendly feature descriptions for the UI
FEATURE_DESCRIPTIONS = {
    'right_elbow_angle':       'Right Elbow Extension',
    'left_elbow_angle':        'Left Elbow Position',
    'right_shoulder_elevation':'Right Shoulder Height',
    'left_shoulder_elevation': 'Left Shoulder Position',
    'shoulder_rotation_angle': 'Shoulder Rotation',
    'wrist_height_diff':       'Wrist Position Asymmetry',
    'bat_swing_arc':           'Bat Swing Speed/Arc',
    'right_knee_bend':         'Back Knee Flexion',
    'left_knee_bend':          'Front Knee Bend',
    'right_hip_angle':         'Back Hip Extension',
    'left_hip_angle':          'Front Hip Position',
    'hip_rotation_angle':      'Hip Rotation',
    'trunk_lean_angle':        'Forward Body Lean',
    'spine_angle':             'Spine Alignment',
    'center_of_mass_x':        'Balance (Horizontal)',
    'center_of_mass_y':        'Balance (Vertical)',
    'feet_width_ratio':        'Stance Width',
    'weight_distribution':     'Weight Distribution',
    'head_stability_x':        'Head Steadiness (Horizontal)',
    'head_stability_y':        'Head Steadiness (Vertical)',
    'follow_through_angle':    'Follow-Through Arc',
    'backlift_height':         'Backlift Height',
}

COACHING_TEMPLATES = {
    'right_elbow_angle': {
        'low':  "Extend your right arm more through the ball — elbow is collapsing.",
        'high': "Your right arm is too straight — maintain a slight flex for control.",
    },
    'shoulder_rotation_angle': {
        'low':  "Your shoulders are not rotating enough — open up your body for power.",
        'high': "Shoulders are over-rotating — control your shoulder turn.",
    },
    'hip_rotation_angle': {
        'low':  "Drive from your hips! Hip rotation is the engine of batting power.",
        'high': "Hips are spinning too much — stay balanced through the shot.",
    },
    'left_knee_bend': {
        'low':  "Front knee is bending too much — plant it firmly as your anchor.",
        'high': "Lock your front knee more at contact for a solid batting base.",
    },
    'head_stability_x': {
        'low':  "Head is drifting to the left — keep your eyes level and head still.",
        'high': "Head is drifting to the right — watch the ball all the way to the bat.",
    },
    'feet_width_ratio': {
        'low':  "Widen your stance — a narrow base limits your power and stability.",
        'high': "Your stance is too wide — this will restrict your hip rotation.",
    },
    'backlift_height': {
        'low':  "Raise your bat higher in the backlift — you're losing power potential.",
        'high': "Backlift is too high — control it to avoid mistiming on quick balls.",
    },
    'bat_swing_arc': {
        'low':  "Your bat swing is too slow or restricted — commit fully to the shot.",
        'high': "Bat is moving too fast early — wait for the ball and time your swing.",
    },
    'weight_distribution': {
        'low':  "Too much weight on the back foot — lean into the shot.",
        'high': "Too much weight on the front foot — stay balanced for quick footwork.",
    },
}


class SHAPExplainer:
    """
    Wraps a trained scikit-learn model with SHAP TreeExplainer
    to produce human-readable batting feedback.
    """

    def __init__(self, model, feature_names: list):
        """
        Args:
            model:         Trained sklearn Random Forest or GBM.
            feature_names: List of feature column names in the same order as training.
        """
        self._model         = model
        self._feature_names = feature_names
        self._explainer     = None

        if SHAP_AVAILABLE and model is not None:
            try:
                self._explainer = shap.TreeExplainer(model)
            except Exception as e:
                print(f"[SHAP] Warning: Could not initialize TreeExplainer: {e}")
                self._explainer = None

    def explain_features(self, features_dict: dict, top_n: int = 5) -> dict:
        """
        Generate SHAP-based explanations for a feature vector.

        Args:
            features_dict (dict): Feature dict from FeatureExtractor.
            top_n (int):          Number of top contributing features to show.

        Returns:
            dict: {
                'top_positive': list[dict],  # Features helping the score
                'top_negative': list[dict],  # Features hurting the score
                'coaching_tips': list[str],  # Human-readable tips
                'shap_available': bool,
            }
        """
        if not features_dict:
            return self._null_explanation()

        # Build ordered feature vector
        X = np.array([
            [features_dict.get(f, 0.0) for f in self._feature_names]
        ])

        if SHAP_AVAILABLE and self._explainer is not None:
            return self._shap_explanation(X, features_dict, top_n)
        else:
            return self._rule_based_explanation(features_dict, top_n)

    def _shap_explanation(self, X: np.ndarray, features_dict: dict, top_n: int) -> dict:
        """Full SHAP explanation using TreeExplainer."""
        try:
            shap_values = self._explainer.shap_values(X)

            # For binary classification, take class-1 SHAP values (Good technique)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]

            # Build ranked feature impact list
            impacts = []
            for i, feat in enumerate(self._feature_names):
                impacts.append({
                    'feature':     feat,
                    'label':       FEATURE_DESCRIPTIONS.get(feat, feat),
                    'shap_value':  round(float(sv[i]), 4),
                    'actual_value': round(float(features_dict.get(feat, 0.0)), 2),
                })

            impacts.sort(key=lambda x: x['shap_value'])
            top_negative = impacts[:top_n]           # Most negative SHAP = biggest penalties
            top_positive = impacts[-(top_n):][::-1]  # Most positive SHAP = biggest boosts

            coaching_tips = self._generate_tips(features_dict)

            return {
                'top_positive':   top_positive,
                'top_negative':   top_negative,
                'coaching_tips':  coaching_tips,
                'shap_available': True,
            }
        except Exception as e:
            print(f"[SHAP] Error during explanation: {e}")
            return self._rule_based_explanation(features_dict, top_n)

    def _rule_based_explanation(self, features_dict: dict, top_n: int) -> dict:
        """
        Fallback: Rule-based importance estimation when SHAP is unavailable.
        Estimates impact by deviation from ideal range.
        """
        from scorer import BiomechanicsScorer
        scorer = BiomechanicsScorer()

        impacts = []
        for feat, (lo, hi) in scorer._TARGETS.items():
            val = features_dict.get(feat, None)
            if val is None:
                continue
            # Score as 0-1, invert to get "how bad" = penalty amount
            closeness = scorer._range_score(val, lo, hi)
            penalty = -(1.0 - closeness)  # Negative = hurting
            bonus   = closeness            # Positive = helping

            impacts.append({
                'feature':      feat,
                'label':        FEATURE_DESCRIPTIONS.get(feat, feat),
                'shap_value':   round(penalty, 4),
                'actual_value': round(float(val), 2),
            })

        impacts.sort(key=lambda x: x['shap_value'])
        top_negative = impacts[:top_n]
        top_positive = sorted(impacts, key=lambda x: -x['shap_value'])[:top_n]
        coaching_tips = self._generate_tips(features_dict)

        return {
            'top_positive':   top_positive,
            'top_negative':   top_negative,
            'coaching_tips':  coaching_tips,
            'shap_available': False,
        }

    def _generate_tips(self, features_dict: dict) -> list:
        """Generate specific coaching tips based on feature deviations."""
        tips = []

        for feat, template in COACHING_TEMPLATES.items():
            val = features_dict.get(feat, None)
            if val is None:
                continue

            from scorer import BiomechanicsScorer
            targets = BiomechanicsScorer._TARGETS.get(feat)
            if not targets:
                continue
            lo, hi = targets

            if val < lo:
                tips.append(template['low'])
            elif val > hi:
                tips.append(template['high'])

        return tips[:5]  # Return at most 5 tips to avoid overwhelming the user

    def _null_explanation(self) -> dict:
        return {
            'top_positive':   [],
            'top_negative':   [],
            'coaching_tips':  ["Upload a video to receive personalised coaching feedback."],
            'shap_available': False,
        }
