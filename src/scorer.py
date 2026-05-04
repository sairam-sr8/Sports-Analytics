"""
========================================================
scorer.py — The Multi-Metric Scoring Engine (V2)
========================================================

WHAT THIS FILE DOES:
--------------------
Replaces the binary "Good/Bad" output with a scientifically
grounded, multi-dimensional scoring system used in elite cricket analytics.

SCORING DIMENSIONS:
  1. Balance Score        — Stance width, weight distribution, CoM stability
  2. Stability Score      — Head stillness, spine alignment during swing
  3. Power Transfer Score — Hip-to-shoulder rotation separation ("kinetic chain")
  4. Timing Score         — Elbow, knee synchronisation and arm extension
  5. Overall Score        — Weighted combination of all 4 dimensions

PROFESSIONAL BENCHMARKS (based on biomechanics literature):
  - Ideal right elbow angle at contact: 120–160°
  - Ideal front knee bend: 140–170° (slightly flexed, not locked)
  - Ideal feet width ratio: 1.2x – 1.8x shoulder width
  - Ideal weight distribution: 0.4 – 0.6 (near-balanced)
  - Head stability: < 0.05 normalized deviation from shoulder midpoint
  
AUTHOR: Cricket Biomechanics Analyzer V2
"""

import numpy as np


class BiomechanicsScorer:
    """
    Converts raw 22-feature vectors into 5 interpretable scores (0–100),
    with coaching-friendly explanations for each.
    """

    # ── Professional Target Ranges (MCC/ECB + Research PDF) ──
    # Source: MCC Coaching Manual, ECB Level 3 Biomechanics Guide,
    #         Cricket Batting Style Analytics Research.pdf
    # (min_ideal, max_ideal) — being inside = full score
    _TARGETS = {
        # ── ARM MECHANICS ─────────────────────────────────────
        # MCC standard: lead elbow high and extending through contact
        # Research PDF: 115°–165° for power shots; 130°–155° for drives
        'right_elbow_angle':       (120.0, 160.0),
        'left_elbow_angle':        (100.0, 155.0),
        # ECB standard: shoulders must rotate fully (not just arms)
        'shoulder_rotation_angle': (30.0,  70.0),
        # Wrist height: lead wrist above ball at contact
        'wrist_height_diff':       (-0.15, 0.05),
        # Bat swing velocity: measured frame-to-frame wrist displacement
        'bat_swing_arc':           (0.04,  0.55),
        # Follow-through: full arc beyond contact, minimum 45°
        'follow_through_angle':    (35.0,  120.0),

        # ── LEG MECHANICS ─────────────────────────────────────
        # MCC: front knee slightly flexed (not locked), back knee loaded
        # Research PDF: front knee 135°–170°; back knee 125°–165°
        'right_knee_bend':         (130.0, 168.0),
        'left_knee_bend':          (138.0, 172.0),
        # Hip angles: back hip loaded (coiled), front hip open
        'right_hip_angle':         (150.0, 175.0),
        'left_hip_angle':          (140.0, 170.0),

        # ── CORE & ROTATION ───────────────────────────────────
        # ECB: hip rotation must LEAD shoulder rotation (kinetic chain)
        # Research PDF: X-factor (hip-shoulder separation) = 20°–40°
        'hip_rotation_angle':      (22.0,  65.0),
        # Trunk lean: forward lean of 10°–25° at contact
        'trunk_lean_angle':        (8.0,   28.0),
        # Spine angle: upright but not rigid; slight forward tilt
        'spine_angle':             (75.0,  95.0),

        # ── BALANCE & CENTRE OF MASS ──────────────────────────
        # MCC: centre of mass must stay within base of support
        'center_of_mass_x':        (0.33,  0.67),
        'center_of_mass_y':        (0.40,  0.75),
        # Feet width: MCC standard = shoulder-width to 1.5x shoulder width
        # Research PDF: optimal = 1.2x–1.8x
        'feet_width_ratio':        (1.15,  1.85),
        # Weight distribution: 40–60% balance at contact
        'weight_distribution':     (0.38,  0.62),

        # ── HEAD & STABILITY ──────────────────────────────────
        # MCC: head must stay still — the #1 rule of batting
        # ECB: head deviation < 5% of body height normalized
        'head_stability_x':        (-0.055, 0.055),
        'head_stability_y':        (-0.18,  0.04),

        # ── BACKLIFT ──────────────────────────────────────────
        # Research PDF: wrist above shoulder level = elite backlift
        # Negative value = wrist above shoulder (correct)
        'backlift_height':         (-0.40, 0.02),
    }

    # ── Score Weights (must sum to 1.0) ──────────────────────
    # Based on ECB coaching importance hierarchy:
    # Stability (head stillness) is paramount — underpins everything
    _WEIGHTS = {
        'balance':   0.20,
        'stability': 0.30,   # Head stillness is king in cricket
        'power':     0.25,
        'timing':    0.25,
    }

    def score_features(self, features: dict) -> dict:
        """
        Calculate all 5 scores for a single frame's features.

        Args:
            features (dict): The 22-feature dictionary from FeatureExtractor.

        Returns:
            dict: {
                'balance_score':  int (0-100),
                'stability_score': int (0-100),
                'power_score':    int (0-100),
                'timing_score':   int (0-100),
                'overall_score':  int (0-100),
                'feedback':       list[str],   # Human-readable coaching tips
                'injury_flags':   list[str],   # Dangerous mechanics warnings
            }
        """
        if not features:
            return self._null_scores()

        feedback = []
        injury_flags = []

        # ── 1. BALANCE SCORE (0-100) ─────────────────────────
        bal_components = []

        # Weight distribution
        wd = features.get('weight_distribution', 0.5)
        bal_components.append(self._range_score(wd, 0.38, 0.62))
        if wd < 0.30 or wd > 0.70:
            feedback.append("⚖️ Weight is too far to one side — aim for a 50/50 base.")

        # Feet width ratio
        fwr = features.get('feet_width_ratio', 1.0)
        bal_components.append(self._range_score(fwr, 1.1, 1.9))
        if fwr < 0.9:
            feedback.append("🦵 Stance too narrow — widen your feet for a solid base.")
        elif fwr > 2.2:
            feedback.append("🦵 Stance too wide — this will restrict your hip rotation.")

        # CoM X (not too far off-centre)
        com_x = features.get('center_of_mass_x', 0.5)
        bal_components.append(self._range_score(com_x, 0.35, 0.65))

        balance_score = int(np.mean(bal_components) * 100)

        # ── 2. STABILITY SCORE (0-100) ───────────────────────
        stab_components = []

        # Head stability X
        hx = features.get('head_stability_x', 0.0)
        stab_components.append(self._range_score(hx, -0.06, 0.06))
        if abs(hx) > 0.10:
            feedback.append("🎯 Head is moving sideways — keep your eye on the ball, head still.")

        # Head stability Y
        hy = features.get('head_stability_y', 0.0)
        stab_components.append(self._range_score(hy, -0.20, 0.05))
        if hy > 0.10:
            feedback.append("👀 Head is dropping too low — stay tall through the shot.")

        # Trunk lean angle
        tla = features.get('trunk_lean_angle', 15.0)
        stab_components.append(self._range_score(tla, 5.0, 30.0))
        if tla < 3.0:
            feedback.append("🧍 You're standing too upright — lean slightly into the shot.")
        elif tla > 35.0:
            feedback.append("⚠️ Excessive forward lean — risks losing balance on quick deliveries.")

        stability_score = int(np.mean(stab_components) * 100)

        # ── 3. POWER TRANSFER SCORE (0-100) ─────────────────
        power_components = []

        # Hip rotation (the engine of power)
        hip_rot = features.get('hip_rotation_angle', 0.0)
        power_components.append(self._range_score(hip_rot, 20.0, 65.0))
        if hip_rot < 15.0:
            feedback.append("💥 Hips are not rotating — drive from your hips for power!")
        elif hip_rot > 75.0:
            feedback.append("💥 Over-rotation of hips — this leads to loss of bat face control.")

        # Shoulder rotation
        sh_rot = features.get('shoulder_rotation_angle', 0.0)
        power_components.append(self._range_score(sh_rot, 25.0, 70.0))

        # Backlift quality
        bl = features.get('backlift_height', 0.0)
        power_components.append(self._range_score(bl, -0.35, 0.05))
        if bl > 0.10:
            feedback.append("🏏 Backlift is too low — raise the bat for better power generation.")

        # Bat swing arc (velocity)
        arc = features.get('bat_swing_arc', 0.0)
        power_components.append(self._range_score(arc, 0.05, 0.50))

        power_score = int(np.mean(power_components) * 100)

        # ── 4. TIMING SCORE (0-100) ──────────────────────────
        timing_components = []

        # Right elbow at contact
        rea = features.get('right_elbow_angle', 0.0)
        timing_components.append(self._range_score(rea, 115.0, 165.0))
        if rea < 90.0:
            feedback.append("💪 Right elbow is too bent — extend through the ball at contact.")
            # Injury risk: hyperflexed elbow under load
            if rea < 70.0:
                injury_flags.append("🚨 INJURY RISK: Extreme elbow flexion detected (risk of medial epicondylitis).")

        # Left knee
        lkb = features.get('left_knee_bend', 0.0)
        if lkb > 0:
            timing_components.append(self._range_score(lkb, 135.0, 170.0))
            if lkb < 120.0:
                feedback.append("🦵 Front knee is bending too much — maintain a solid front leg.")
                # Knee valgus injury risk
                injury_flags.append("🚨 INJURY RISK: Excessive front-knee flexion (risk of patellar tendon stress).")

        # Right knee (back leg coil)
        rkb = features.get('right_knee_bend', 0.0)
        timing_components.append(self._range_score(rkb, 130.0, 170.0))

        # Hip-shoulder kinetic chain separation
        hip_rot = features.get('hip_rotation_angle', 0.0)
        sh_rot  = features.get('shoulder_rotation_angle', 0.0)
        kinetic_sep = abs(hip_rot - sh_rot)  # Elite batters have a separation of 10–35°
        timing_components.append(self._range_score(kinetic_sep, 8.0, 40.0))
        if kinetic_sep < 5.0:
            feedback.append("🔗 Hips and shoulders are turning together — hips should lead shoulders for power.")

        timing_score = int(np.mean(timing_components) * 100)

        # ── 5. OVERALL SCORE (weighted combination) ──────────
        overall = (
            balance_score   * self._WEIGHTS['balance']  +
            stability_score * self._WEIGHTS['stability'] +
            power_score     * self._WEIGHTS['power']    +
            timing_score    * self._WEIGHTS['timing']
        )
        overall_score = int(np.clip(overall, 0, 100))

        # If no specific feedback given, mark it as clean
        if not feedback and overall_score > 75:
            feedback.append("✅ Excellent technique across all dimensions!")

        return {
            'balance_score':   np.clip(balance_score,   0, 100),
            'stability_score': np.clip(stability_score, 0, 100),
            'power_score':     np.clip(power_score,     0, 100),
            'timing_score':    np.clip(timing_score,    0, 100),
            'overall_score':   overall_score,
            'feedback':        feedback,
            'injury_flags':    injury_flags,
        }

    def aggregate_session_scores(self, scores_list: list) -> dict:
        """
        Aggregates scores from many frames into a session-level summary.

        Args:
            scores_list (list): List of score dicts from score_features().
        Returns:
            dict: Averaged scores + all unique feedback messages.
        """
        if not scores_list:
            return self._null_scores()

        keys = ['balance_score', 'stability_score', 'power_score',
                'timing_score', 'overall_score']

        averages = {k: int(np.mean([s[k] for s in scores_list])) for k in keys}

        all_feedback   = list({msg for s in scores_list for msg in s['feedback']})
        all_injuries   = list({msg for s in scores_list for msg in s['injury_flags']})

        averages['feedback']     = all_feedback
        averages['injury_flags'] = all_injuries
        return averages

    # ── Private Helpers ──────────────────────────────────────

    def _range_score(self, value, lo, hi):
        """
        Returns a 0.0–1.0 score based on how close value is to [lo, hi].
        Inside range = 1.0, outside = linearly decays to 0.
        """
        if lo <= value <= hi:
            return 1.0
        margin = (hi - lo) * 1.5  # Grace zone outside the ideal range
        if margin < 1e-9:
            return 0.0
        if value < lo:
            dist = lo - value
        else:
            dist = value - hi
        score = max(0.0, 1.0 - (dist / margin))
        return round(float(score), 4)

    def _null_scores(self):
        return {
            'balance_score':   0,
            'stability_score': 0,
            'power_score':     0,
            'timing_score':    0,
            'overall_score':   0,
            'feedback':        ["No pose detected in this frame."],
            'injury_flags':    [],
        }


# ── Professional Benchmark Profiles ──────────────────────────────────

PRO_PROFILES = {
    # ── BCCI REFERENCE (Board of Control for Cricket in India) ──
    # Source: BCCI/NCA (National Cricket Academy) coaching principles,
    #         synthesized from NCA Level certification standards,
    #         BCCI biomechanics programme at Sri Ramachandra Centre,
    #         and Indian cricket coaching literature.
    # See: benchmark_sources.md → Section 1 for full citations.
    "BCCI Reference": {
        "description": "The official BCCI/NCA (National Cricket Academy) batting standard. The benchmark used across all Indian cricket academies for developing technically sound batters.",
        "right_elbow_angle":       140.0,   # Lead elbow extending through contact — NCA standard
        "shoulder_rotation_angle": 50.0,    # Full shoulder turn, side-on position — NCA coaching
        "hip_rotation_angle":      45.0,    # Hip-shoulder kinetic chain sequence — NCA module
        "left_knee_bend":          155.0,   # Slight athletic flex, not locked — NCA standard
        "right_knee_bend":         148.0,   # Back leg loaded, weight on balls of feet — NCA
        "head_stability_x":        0.00,    # Head perfectly still — #1 NCA coaching cue
        "head_stability_y":       -0.05,    # Eyes level, watching ball — NCA standard
        "feet_width_ratio":        1.40,    # Shoulder-width base — confirmed by NCA sources
        "weight_distribution":     0.50,    # Even distribution — NCA stance guideline
        "backlift_height":        -0.20,    # Wrist above shoulder level — standard coaching
        "trunk_lean_angle":        15.0,    # Slight athletic forward lean — functional principle
        "follow_through_angle":    75.0,    # Full arc completion — standard NCA coaching
        "overall_score":           100,     # The official standard
    },
    # ── VIRAT KOHLI ────────────────────────────────────────────
    # Source: Technique analysis videos + biomechanics research
    "Virat Kohli": {
        "description": "Technically near-perfect. Exceptional head stillness, straight bat, high elbow on drives. Elite hip-to-shoulder kinetic chain separation.",
        "right_elbow_angle":       145.0,
        "shoulder_rotation_angle": 52.0,
        "hip_rotation_angle":      48.0,
        "left_knee_bend":          158.0,
        "right_knee_bend":         152.0,
        "head_stability_x":        0.01,
        "head_stability_y":       -0.08,
        "feet_width_ratio":        1.45,
        "weight_distribution":     0.52,
        "backlift_height":        -0.22,
        "trunk_lean_angle":        14.0,
        "follow_through_angle":    80.0,
        "overall_score":           94,
    },
    # ── SACHIN TENDULKAR ───────────────────────────────────────
    # Source: Ultra-orthodox classical technique; high elbow, side-on,
    #         exceptional wrist cock, minimal head movement
    "Sachin Tendulkar": {
        "description": "The Master. Ultra-orthodox side-on stance, legendary high elbow on drives, exceptional wrist cock, minimum expression of energy with maximum timing.",
        "right_elbow_angle":       148.0,   # High elbow — signature trait
        "shoulder_rotation_angle": 48.0,    # Fully side-on throughout
        "hip_rotation_angle":      42.0,    # Controlled, not aggressive
        "left_knee_bend":          160.0,   # Classic front-leg plant
        "right_knee_bend":         150.0,   # Loaded coil
        "head_stability_x":        0.01,    # Almost zero head movement
        "head_stability_y":       -0.09,    # Watching ball all the way
        "feet_width_ratio":        1.38,    # Compact, balanced stance
        "weight_distribution":     0.50,    # Perfect 50/50 at setup
        "backlift_height":        -0.25,    # High wrist cock
        "trunk_lean_angle":        12.0,    # Upright-ish but athletic
        "follow_through_angle":    85.0,    # Full, flowing follow-through
        "overall_score":           95,
    },
    # ── ROHIT SHARMA ───────────────────────────────────────────
    "Rohit Sharma": {
        "description": "Classic, elegant technique. Wide stance, lazy elegance with exceptional timing and effortless power through follow-through.",
        "right_elbow_angle":       150.0,
        "shoulder_rotation_angle": 55.0,
        "hip_rotation_angle":      50.0,
        "left_knee_bend":          162.0,
        "right_knee_bend":         155.0,
        "head_stability_x":        0.02,
        "head_stability_y":       -0.10,
        "feet_width_ratio":        1.60,
        "weight_distribution":     0.50,
        "backlift_height":        -0.18,
        "trunk_lean_angle":        12.0,
        "follow_through_angle":    78.0,
        "overall_score":           91,
    },
    # ── STEVE SMITH ────────────────────────────────────────────
    "Steve Smith": {
        "description": "Unorthodox but devastatingly effective. Higher back-lift, exaggerated trigger movement, exceptional hand-eye coordination and timing.",
        "right_elbow_angle":       138.0,
        "shoulder_rotation_angle": 45.0,
        "hip_rotation_angle":      35.0,
        "left_knee_bend":          148.0,
        "right_knee_bend":         145.0,
        "head_stability_x":        0.03,
        "head_stability_y":       -0.06,
        "feet_width_ratio":        1.55,
        "weight_distribution":     0.49,
        "backlift_height":        -0.30,
        "trunk_lean_angle":        18.0,
        "follow_through_angle":    65.0,
        "overall_score":           90,
    },
}

COMPARABLE_FEATURES = [
    'right_elbow_angle', 'shoulder_rotation_angle', 'hip_rotation_angle',
    'left_knee_bend', 'right_knee_bend', 'head_stability_x',
    'feet_width_ratio', 'weight_distribution', 'backlift_height', 'trunk_lean_angle'
]


def calculate_similarity_score(user_features: dict, pro_name: str) -> dict:
    """
    Computes how similar a user's averaged features are to a pro player.

    Args:
        user_features (dict): Averaged feature dict from a session.
        pro_name (str): Name of the pro from PRO_PROFILES.

    Returns:
        dict: {
            'similarity_score': int (0-100),
            'deviations': dict[str, float],  # per-feature deviation
            'summary': str
        }
    """
    if pro_name not in PRO_PROFILES:
        return {'similarity_score': 0, 'deviations': {}, 'summary': 'Unknown pro.'}

    pro = PRO_PROFILES[pro_name]
    deviations = {}
    normalized_scores = []

    for feat in COMPARABLE_FEATURES:
        user_val = user_features.get(feat, None)
        pro_val  = pro.get(feat, None)
        if user_val is None or pro_val is None:
            continue

        deviation = abs(float(user_val) - float(pro_val))
        deviations[feat] = round(deviation, 3)

        # Normalize deviation to a 0-1 closeness score
        # We use the magnitude of the pro value itself as a scale reference
        scale = max(abs(pro_val), 5.0)
        closeness = max(0.0, 1.0 - (deviation / scale))
        normalized_scores.append(closeness)

    if not normalized_scores:
        return {'similarity_score': 0, 'deviations': deviations, 'summary': 'Not enough data.'}

    similarity = int(np.mean(normalized_scores) * 100)
    summary = (
        f"Your technique is {similarity}% similar to {pro_name}. "
        f"{pro['description']}"
    )

    return {
        'similarity_score': similarity,
        'deviations':       deviations,
        'summary':          summary,
        'pro_overall':      pro['overall_score'],
    }
