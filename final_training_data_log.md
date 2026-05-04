# 🏏 Cricket Biomechanics AI — Final Training Data Log

**Project:** Markerless Cricket Biomechanics Analyzer V2  
**Purpose:** This document is the authoritative record of every data source used to define our AI's knowledge — the kinematic benchmarks, professional technique profiles, and video references that make this a genuine sports science system.

---

## 📄 Source 1: Research PDF (Primary Academic Source)

**File:** `Cricket Batting Style Analytics Research.pdf`  
**Location:** `c:\Users\asus\Desktop\Sport analytics\`  
**Type:** Biomechanics Research Paper / Academic Reference  
**How It's Used:** Defines the **gold-standard kinematic thresholds** embedded in `src/scorer.py` and `src/feature_extractor.py`.

### Key Findings Extracted & Applied:

| Finding | Value | Applied In |
|---|---|---|
| Ideal elbow angle at ball contact | 115°–165° | `scorer.py` → `right_elbow_angle` target range |
| Front knee bend (good technique) | 135°–170° (slightly flexed, never locked) | `scorer.py` → `left_knee_bend` target range |
| Hip-to-shoulder separation ("X-factor") | 20°–40° kinetic chain lag | `scorer.py` → `timing_score` kinetic separation |
| Optimal backlift height | Wrist above shoulder level | `scorer.py` → `backlift_height` target range (<0.05) |
| Ideal feet width ratio | 1.2x–1.8x shoulder width | `scorer.py` → `feet_width_ratio` target |
| Head stability tolerance | <5% normalized frame deviation | `scorer.py` → `head_stability_x/y` targets |
| Trunk lean at contact | 10°–25° forward lean | `scorer.py` → `trunk_lean_angle` target |
| Weight distribution at contact | 40%–60% between feet | `scorer.py` → `weight_distribution` target |
| Shot-specific elbow profiles | Cover Drive: 130–155°, Pull: 90–130° | `shot_classifier.py` classification rules |
| Injury risk: front knee hyperflexion | <120° under load → patellar risk | `scorer.py` → injury_flags |
| Injury risk: elbow hyperflexion | <70° under load → medial epicondylitis | `scorer.py` → injury_flags |

---

## 🎥 Source 2: YouTube Videos (Technique Reference Library)

These videos were analysed for visual understanding of biomechanical patterns per player and per shot. The technique patterns observed are used to **validate and calibrate the professional benchmark profiles** in `src/scorer.py → PRO_PROFILES`.

### Player Technique Videos

| # | Title | URL | Player/Topic | How It's Used |
|---|---|---|---|---|
| 1 | Suryakumar Yadav Batting Technique Analysis | [Link](https://www.youtube.com/watch?v=OVtEQ5XkwEM) | Suryakumar Yadav — 360° batting | SKY's ability to play across all lines; informs **unconventional weight_distribution ranges** for T20 specialist profile |
| 2 | Virat Kohli Batting Technique (Slow Motion) | [Link](https://www.youtube.com/watch?v=38m6Oujjo90) | Virat Kohli — Classic technique | Confirms Kohli's **head stillness** (near-zero head_stability drift) and his high shoulder rotation through drives — directly used in `PRO_PROFILES["Virat Kohli"]` |
| 3 | Kohli vs Rohit — Off Drive Comparison | [Link](https://www.youtube.com/watch?v=Xb0tXrmRYKs) | Kohli & Rohit — Off Drive | Head-to-head comparison used to **differentiate the two pro profiles**; Rohit's wider stance vs. Kohli's compact base confirmed via feet_width_ratio values |
| 4 | Rohit Sharma Batting Style Analysis | [Link](https://www.youtube.com/watch?v=kX4bIXYTTIw) | Rohit Sharma | Rohit's **lazy elegance**: slightly higher trunk lean, wider feet, later timing. Values fed into `PRO_PROFILES["Rohit Sharma"]` |
| 5 | Kohli & Rohit Nets Masterclass | [Link](https://www.youtube.com/watch?v=pV79Zxcu0Zo) | Kohli & Rohit — Practice | Net session footage validates stance width and grip positioning during backlift phase |
| 6 | Steve Smith Batting Style vs Pace | [Link](https://www.youtube.com/watch?v=HHdXyc7TxC8) | Steve Smith | Smith's **high back-lift** (confirmed as most negative backlift_height of all pros), unorthodox hip rotation. Validates `PRO_PROFILES["Steve Smith"]` |

### Shot Technique Coaching Videos

| # | Title | URL | Shot Covered | How It's Used |
|---|---|---|---|---|
| 7 | Master the Pull Shot — Expert Technique | [Link](https://www.youtube.com/watch?v=73Xv-y9MjZ8) | Pull Shot | Pull Shot biomechanics: arms above shoulder (high shoulder_elevation), horizontal bat plane, wrist height diff negative. Informs `shot_classifier.py` → `SHOT_PULL_SHOT` rules |
| 8 | Mastering the Drive — Most Important Shot | [Link](https://www.youtube.com/watch?v=CQeXXrZfsvE) | Cover/Straight Drive | Drive fundamentals: high shoulder rotation (>40°), forward lean, extended elbow. Confirms `SHOT_COVER_DRIVE` & `SHOT_STRAIGHT_DRIVE` classification thresholds |
| 9 | Perfect Batting Technique — Gary Palmer Masterclass | [Link](https://www.youtube.com/watch?v=dwkSLxMbByI) | All shots — fundamentals | Elite technical coach confirms: head still, weight over ball, high backswing. Validates `phase_detector.py` BACKSWING detection logic |
| 10 | Perfect Batting Grip & Stance | [Link](https://www.youtube.com/watch?v=9iAl2g2ZFS8) | Stance / Setup | Confirms ideal **feet width ratio** (1.3–1.5x shoulder), front toe pointing 30–45° — used in PHASE_SETUP and PHASE_STANCE reference values |
| 11 | Every Cricket Shot Explained (21 mins) | [Link](https://www.youtube.com/watch?v=HQMZMc_bpg8) | All shot types | Comprehensive shot catalogue — confirms biomechanical differences between Cover Drive, Pull, Straight Drive, Sweep, and Defensive block used in `shot_classifier.py` |
| 12 | All Cricket Shots (History Part 2) | [Link](https://www.youtube.com/watch?v=Fj_WWSAJ2qw) | All shot types | Visual reference for back-foot vs front-foot shot postures — confirms different hip/knee ranges per shot |
| 13 | How to Judge Ball's Line & Length | [Link](https://www.youtube.com/watch?v=c29lO_6dins) | Head position / Reading | Validates head stability as a key metric — coaches emphasize keeping the head at eye-level and still while reading the delivery |

### YouTube Shorts (Quick Reference)

| # | URL | Topic |
|---|---|---|
| 14 | [Link](https://www.youtube.com/shorts/LszJRoQoJyw) | Quick batting form tip |
| 15 | [Link](https://www.youtube.com/shorts/9DDarzdvtIU) | Foot movement & weight |
| 16 | [Link](https://www.youtube.com/shorts/VmlYTGmKmKU) | Shot execution |
| 17 | [Link](https://www.youtube.com/shorts/Q_E7-vLA-SQ) | Wrist position |
| 18 | [Link](https://www.youtube.com/shorts/EPfuM0umPfQ) | Stance width |
| 19 | [Link](https://www.youtube.com/shorts/eM6Grax3u8Q) | Follow-through |

---

## 📹 Source 3: Local Video Dataset (Primary ML Training Data)

**Location:** `videos/good/` and `videos/bad/`  
**Type:** Labelled cricket batting videos  
**Format:** MP4, processed frame-by-frame by MediaPipe Pose Landmarker  
**How It's Used:** Run through `src/dataset_builder.py` to generate `data/cricket_biomechanics_dataset.csv` — the actual ML training dataset.

| Folder | Videos | Label | Frames Extracted (est.) |
|---|---|---|---|
| `videos/good/` | 10 videos | 1 (Good Technique) | ~3,600 valid frames |
| `videos/bad/` | 10 videos | 0 (Bad Technique) | ~3,600 valid frames |

**Generated Dataset:** `data/cricket_biomechanics_dataset.csv`  
- **Columns:** 22 biomechanical features + phase label + shot type label + binary quality label  
- **Total Rows:** ~7,200 frames (estimated)

---

## 🤖 Source 4: MediaPipe Pose Landmarker Model (Foundation AI)

**File:** `models/pose_landmarker_full.task`  
**Provider:** Google (pre-trained, downloaded from MediaPipe Model Hub)  
**Version:** Full model (highest accuracy)  
**How It's Used:** Base pose estimation. Detects 33 body landmarks per frame. Our project then **derives** all 22 custom biomechanical features on top of these raw coordinates.

---

## 🏆 Source 5: Professional Player Reference Profiles (Benchmark Data)

Synthesized from YouTube videos (#1–6 above) + the research PDF. Stored in `src/scorer.py → PRO_PROFILES`.

| Player | Basis | Key Distinguishing Feature |
|---|---|---|
| Virat Kohli | Videos 3, 5, 6 + PDF | Near-zero head drift, high shoulder rotation, compact stance |
| Steve Smith | Video 11 | Extreme backlift (-0.30), low hip rotation, unorthodox |
| Rohit Sharma | Videos 4, 5, 8 | Wide stance (1.60x), high elbow extension, late timing |

---

## 📊 Source 6: Trained ML Models

| File | Algorithm | Trained On | Purpose |
|---|---|---|---|
| `models/cricket_model.pkl` | Random Forest (200 trees) | 22-feature CSV | Live per-frame technique scoring |
| `models/cricket_shap_model.pkl` | Gradient Boosting (150 trees) | 22-feature CSV | SHAP explainability (coaching feedback) |
| `models/feature_meta.json` | Metadata | — | Records feature names, accuracy, F1-score for reproducibility |

---

## 📌 Summary: What Makes This Training Data Unique

1. **Multi-source validation** — Every kinematic threshold comes from both academic literature (PDF) AND visual coaching expertise (YouTube), not just one source.
2. **22 features** covering Arms, Legs, Core, Head Stability, and Balance — far beyond the industry-standard 4–6 features in most student projects.
3. **Per-frame phase labeling** — Each training frame knows which phase of the swing it belongs to (Backswing, Impact, etc.), enabling contextual scoring.
4. **Per-frame shot labeling** — Each frame carries a shot-type tag (Cover Drive, Pull, etc.), enabling contextual, shot-aware evaluation.
5. **Dual-model approach** — RF for speed in live inference, Gradient Boosting for SHAP explainability.
