# Cricket Biomechanics AI — Full Project Documentation
### Version 2.0 | Enterprise Sports Analytics

---

## TABLE OF CONTENTS
1. What We Built & Why
2. Who This Is For
3. The Big Picture — How It Works (Non-Technical)
4. Tech Stack
5. Project Architecture — Every File Explained
6. The Data Pipeline (Eyes of the AI)
7. The ML Brain — Models We Used & Why
8. The Scoring Engine
9. The Web App — What Happens When You Upload a Video
10. The Live Camera Mode
11. The PDF Coaching Report
12. The Database & Player Progress
13. Pro Comparison System
14. Setup Guide — Recreate Everything From Scratch
15. GitHub & Deployment
16. Future Roadmap

---

## 1. WHAT WE BUILT & WHY

### What is it?
The **Markerless Cricket Biomechanics Analyzer** is an AI-powered sports analytics web application that analyzes a cricket batter's body movements using only a standard smartphone camera — no special sensors, no expensive lab equipment, no stick-on markers required.

### The Problem We Solved
Professional cricket teams (BCCI, ECB, Cricket Australia) use expensive motion-capture labs costing ₹50–₹200 lakhs to analyze a batter's technique. Grassroots coaches, school teams, and academy players have zero access to this technology. A batting coach can only watch with their eyes — they cannot measure joint angles, rotation speeds, or biomechanical risk factors.

### Our Solution
Using a smartphone camera + AI, we automatically:
- Detect 33 skeletal keypoints on the batter's body in every single frame
- Calculate 22 precise biomechanical measurements (joint angles, rotation, balance)
- Score the technique across 4 dimensions: Balance, Stability, Power, Timing
- Compare the batter's technique to Virat Kohli, Sachin Tendulkar, and other pros
- Generate a professional PDF coaching report with SHAP AI explanations
- Detect injury risk patterns before they cause damage

### Why Does This Matter?
- A cricket coach at a small academy can now give data-driven feedback like "your hip-shoulder separation is 12 degrees — it needs to be 35+ for elite power transfer"
- Parents investing in coaching can see measurable improvement over time
- Players can self-analyze using their own phone camera

---

## 2. WHO THIS IS FOR

| Audience | Value |
|---|---|
| **Player** | Objective feedback on every shot, track improvement over months |
| **Coach** | Data-driven coaching, injury prevention alerts, PDF reports to share |
| **Academy** | Affordable pro-grade biomechanics for every student |
| **Investor** | Scalable SaaS model — any sport, any country |
| **Researcher** | 22-feature dataset with augmented training data |

---

## 3. THE BIG PICTURE — HOW IT WORKS (NON-TECHNICAL)

Think of our system in two parts: **Eyes** and a **Brain**.

### The Eyes — MediaPipe Pose Detection
When you upload a video of a batter, the computer watches every single frame (30 frames per second). For each frame, it finds the batter's body and places 33 invisible "dots" on key body parts: nose, shoulders, elbows, wrists, hips, knees, ankles.

These 33 dots are called **landmarks**. They give us the (x, y) coordinates of every joint.

### The Brain — Machine Learning
Once we have those joint positions, we calculate 22 measurements. For example:
- The angle between shoulder → elbow → wrist (elbow angle)
- The distance between the feet compared to shoulder width (stance width)
- How much the hips have rotated vs the shoulders (kinetic chain)

We then compare these 22 measurements against two things:
1. **Rule-Based Scoring** — MCC/ECB professional coaching standards
2. **Trained ML Models** — Patterns learned from 165+ cricket videos

The result is a score out of 100 across 4 dimensions, instant coaching tips, injury risk alerts, and a PDF report.

---

## 4. TECH STACK

| Technology | Purpose | Why We Chose It |
|---|---|---|
| **Python 3.x** | Core language | Scientific libraries, AI support |
| **MediaPipe 0.10.9** | Skeletal pose detection (The Eyes) | Google's production-grade, runs on CPU, no GPU needed |
| **OpenCV (cv2)** | Video frame processing, drawing skeleton | Industry standard for computer vision |
| **scikit-learn** | Random Forest ML model (The Brain) | Fast, interpretable, no GPU needed |
| **SHAP** | AI explainability (why did it give that score?) | Makes AI decisions understandable to coaches |
| **Streamlit** | Web application framework | Rapid deployment, Python-native |
| **streamlit-webrtc** | Live camera WebRTC streaming | Real-time browser camera access |
| **fpdf2** | PDF coaching report generation | Pure Python, no external dependencies |
| **SQLite** | Player session history database | Lightweight, zero-configuration |
| **pandas / numpy** | Data manipulation and math | Scientific computing standard |
| **yt-dlp** | Downloading training videos from YouTube | Free, reliable video downloader |
| **seaborn / matplotlib** | Charts in progress dashboard | Publication-quality visualizations |

---

## 5. PROJECT ARCHITECTURE — EVERY FILE EXPLAINED

```
Sport analytics/
│
├── app.py                          ← THE MAIN WEB APP (orchestrator)
│
├── src/                            ← THE AI ENGINE
│   ├── pose_detector.py            ← The Eyes (MediaPipe wrapper)
│   ├── feature_extractor.py        ← Calculates all 22 biomechanical measurements
│   ├── scorer.py                   ← Converts measurements into scores 0-100
│   ├── phase_detector.py           ← Detects which phase of the shot (Stance, Backlift, etc.)
│   ├── shot_classifier.py          ← Classifies what shot is being played
│   ├── shap_explainer.py           ← AI explanations using SHAP values
│   ├── ml_trainer.py               ← Trains the main Random Forest + SHAP GB model
│   ├── shot_ml_trainer.py          ← Trains the shot classification model
│   ├── dataset_builder.py          ← Processes all training videos into a CSV dataset
│   ├── player_db.py                ← SQLite database for player session history
│   └── report_generator.py         ← Generates the PDF coaching report
│
├── scripts/
│   ├── video_augmenter.py          ← Creates flipped/darkened copies of training videos
│   ├── download_training_videos.py ← Downloads videos from YouTube for training
│   └── diagnostic.py               ← System health check script
│
├── models/
│   ├── cricket_model.pkl           ← Trained Random Forest model (Good/Bad classifier)
│   ├── cricket_shap_model.pkl      ← Trained Gradient Boosting model (for SHAP explanations)
│   ├── shot_classifier.pkl         ← Trained shot classification model
│   ├── feature_meta.json           ← List of feature column names + model accuracy stats
│   └── pose_landmarker_full.task   ← Google MediaPipe pose model file
│
├── data/
│   ├── cricket_biomechanics_dataset.csv  ← The master training dataset (34MB, 165+ videos)
│   └── player_history.db                 ← SQLite database of all recorded sessions
│
├── videos/
│   ├── good/                       ← Training videos of GOOD batting technique
│   ├── bad/                        ← Training videos of POOR batting technique
│   └── samples/
│       └── test_batting_sample.mp4 ← Demo video for testing the app
│
├── reports/                        ← Auto-generated PDF coaching reports
├── logs/                           ← Training and runtime logs
│
├── run_dataset_builder.py          ← Entry point: run dataset_builder on all videos
├── run_ml_trainer.py               ← Entry point: train all ML models from dataset
├── run_test.py                     ← System test to verify everything is working
├── download_samples.py             ← Helper to download a sample test video
├── smart_downloader.py             ← Advanced video downloader with error handling
│
├── requirements.txt                ← Python package dependencies
├── packages.txt                    ← System-level dependencies (for Streamlit Cloud)
├── .gitignore                      ← Files excluded from GitHub (large model/data files)
├── README.md                       ← Quick start guide
├── benchmark_sources.md            ← Citations for all professional biomechanics data
└── final_training_data_log.md      ← Log of which videos were used in training
```

---

## 6. THE DATA PIPELINE (EYES OF THE AI)

### Step 1 — Video Collection (`scripts/download_training_videos.py`, `smart_downloader.py`)
We downloaded cricket batting videos from YouTube. Videos were categorized into:
- `videos/good/` — Professional batters with correct technique (Kohli, Root, etc.)
- `videos/bad/` — Common batting mistakes, coaching drill videos showing incorrect form

### Step 2 — Data Augmentation (`scripts/video_augmenter.py`)
To prevent the AI from overfitting (memorizing only the exact videos it saw), we tripled the training data:
- **Original**: 1 video → kept as-is
- **Augmented (flip)**: Mirror image of the video (left-right flip)
- **Augmented (dark)**: Darkened version simulating poor lighting conditions

Result: From ~55 original videos → 165+ training videos.

### Step 3 — Dataset Building (`src/dataset_builder.py`)
This is the most computationally expensive step (took ~5.5 hours!). For every video:
1. Open the video with OpenCV
2. For every frame, run MediaPipe to find the 33 skeleton landmarks
3. Pass landmarks to `feature_extractor.py` to calculate all 22 measurements
4. Label the row as `GOOD` (from good/ folder) or `BAD` (from bad/ folder)
5. Write the row to `data/cricket_biomechanics_dataset.csv`

Final dataset: **34MB CSV file** with thousands of rows, each representing one video frame.

### Step 4 — Model Training (`src/ml_trainer.py`, `src/shot_ml_trainer.py`)
Run `python run_ml_trainer.py` to read the CSV and train two models:
1. **Random Forest** — classifies each frame as Good or Bad technique
2. **Gradient Boosting (SHAP)** — same task but also tells us *which features* caused the score

---

## 7. THE ML BRAIN — MODELS WE USED & WHY

### Model 1: Random Forest Classifier (`models/cricket_model.pkl`)
**What it is**: An ensemble of 100+ decision trees that vote on whether a frame shows good or bad technique.

**Why Random Forest?**
- Does NOT require a GPU (runs on any laptop)
- Handles small-to-medium datasets very well
- Naturally resistant to overfitting via ensemble voting
- Feature importance is built-in (tells us which joints matter most)

**Final Accuracy: 91.17%** (Cross-validated on held-out test set)

**Top 5 most important features (from the model):**
1. `bat_swing_arc` — swing velocity
2. `head_stability_x` — horizontal head movement
3. `trunk_lean_angle` — forward body lean
4. `spine_angle` — spinal alignment
5. `follow_through_angle` — completion of swing

### Model 2: SHAP Gradient Boosting (`models/cricket_shap_model.pkl`)
**What it is**: A Gradient Boosting Classifier paired with SHAP (SHapley Additive exPlanations).

**Why SHAP?**
- Regular ML models are "black boxes" — they give a result but not a reason
- SHAP assigns each feature a positive or negative contribution to the final score
- This is how the app can say "your elbow angle is the biggest reason your score is low"
- SHAP is based on Nobel Prize-winning game theory (Shapley Values)

### Model 3: Shot Classifier (`models/shot_classifier.pkl`)
**What it is**: A Random Forest that classifies the shot type based on body pose.

**Shot Types Detected:**
- Cover Drive
- Pull Shot
- Sweep
- Cut Shot
- Defensive Block
- Unknown

**How it works**: Different shots have different characteristic joint angle combinations. A Pull Shot has high elbow, strong hip rotation. A Defensive Block has minimal swing arc. The model learned these patterns from the training data.

---

## 8. THE SCORING ENGINE (`src/scorer.py`)

### The 22 Features → 4 Scores Formula

The `scorer.py` file contains the professional benchmarks sourced from:
- MCC (Marylebone Cricket Club) Coaching Manual
- ECB (England & Wales Cricket Board) Level 3 Biomechanics Guide
- BCCI/NCA (National Cricket Academy) coaching standards
- Academic research PDFs on cricket biomechanics

**Score Formula:**
```
Balance Score (20% weight):
  - Weight distribution between feet (ideal: 40-60%)
  - Feet width ratio (ideal: 1.2x-1.8x shoulder width)
  - Centre of mass position

Stability Score (30% weight) ← HIGHEST WEIGHT — head stillness is king:
  - Head movement (X axis): must stay < 5% of body width
  - Head movement (Y axis): must not drop
  - Trunk lean angle: ideal 8-28 degrees forward

Power Score (25% weight):
  - Hip rotation angle (ideal: 22-65 degrees)
  - Shoulder rotation angle
  - Backlift height (wrist above shoulder = elite)
  - Bat swing arc velocity

Timing Score (25% weight):
  - Right elbow angle at contact (ideal: 120-160 degrees)
  - Front knee angle (ideal: 138-172 degrees)
  - Kinetic chain separation (hips must lead shoulders by 10-35 degrees)

Overall Score = (Balance×0.20) + (Stability×0.30) + (Power×0.25) + (Timing×0.25)
```

### Injury Risk Detection
The scorer automatically flags dangerous mechanics:
- Elbow angle < 70°: Risk of medial epicondylitis ("Batter's Elbow")
- Front knee < 120°: Risk of patellar tendon stress

---

## 9. THE WEB APP — WHAT HAPPENS WHEN YOU UPLOAD A VIDEO

### File: `app.py` (503 lines — the orchestrator)

**Step-by-step backend walkthrough:**

**Step 1: App loads**
```
app.py loads → calls load_model() →
reads models/feature_meta.json (feature column names) →
loads models/cricket_model.pkl (Random Forest) →
loads models/cricket_shap_model.pkl (SHAP GB model) →
creates PlayerDatabase() connection to data/player_history.db
```

**Step 2: User uploads video**
```
Streamlit file_uploader widget →
saves to a temp .mp4 file on disk →
sets ready = True
```

**Step 3: User clicks ▶ START ANALYSIS**
```
For every frame in the video:
  1. cv2.VideoCapture reads the next frame (BGR image array)
  2. PoseDetector.process_frame():
       → sends frame to MediaPipe
       → gets back 33 landmark coordinates (x, y, z, visibility)
       → applies smoothing filter (5-frame moving average to reduce jitter)
       → draws skeleton lines on the frame using cv2.line()
  3. FeatureExtractor.extract_features():
       → checks if key joints are visible (visibility > 0.45)
       → calculates all 22 biomechanical measurements using trigonometry
       → returns a dict of {feature_name: float_value}
  4. BiomechanicsScorer.score_features():
       → compares each feature to professional target ranges
       → calculates Balance, Stability, Power, Timing scores (0-100)
       → generates coaching feedback strings
       → flags injury risks
  5. PhaseDetector.detect_phase():
       → uses rules to determine current shot phase
       → Stance → Backlift → Downswing → Contact → Follow-Through
  6. ShotClassifier.classify_frame():
       → uses shot_classifier.pkl ML model
       → classifies: Cover Drive / Pull / Sweep / Cut / Block
  7. The UI updates in real-time:
       → Left panel: video frame with skeleton drawn on it
       → Right panel: metric cards update (Overall, Balance, Stability, Power, Timing, Phase, Shot)
```

**Step 4: Session Summary (after all frames processed)**
```
aggregate_session_scores() → averages all frame scores →
SHAPExplainer.explain_features() → identifies top 4 features dragging score down →
calculate_similarity_score() → compares to selected pro (Kohli, Tendulkar etc.) →
PlayerDatabase.save_session() → saves to SQLite →
generate_pdf_report() → creates downloadable PDF →
st.balloons() fires if overall > 70!
```

---

## 10. THE LIVE CAMERA MODE

### How WebRTC Works in This App
`streamlit-webrtc` creates a direct peer-to-peer video connection between the browser and the Streamlit server using the STUN/TURN relay protocol. This means the camera feed never has to leave your network unnecessarily.

### The LiveProcessor Class (inside app.py)
The key to live camera analysis is the `LiveProcessor` class:

```python
class LiveProcessor:
    def recv(self, frame):
        # This function runs 30 times per second in a background thread
        # It receives each camera frame as it arrives
        img = frame.to_ndarray(format="bgr24")  # Convert to OpenCV format
        out, lm = self.detector.process_frame(img, self.ts)  # Find skeleton
        feats = self.extractor.extract_features(lm)  # Calculate 22 features
        scores = self.scorer_eng.score_features(feats)  # Score the features
        # Draw HUD overlay with metrics directly onto the video frame
        # Write frame to disk using cv2.VideoWriter (saves during recording!)
        return av.VideoFrame.from_ndarray(out, format="bgr24")
```

### Why We Write to Disk During Recording (Not After)
This was a critical technical decision. When you click "Stop", Streamlit triggers a page rerender. During that rerender, the background WebRTC thread's objects are destroyed by Python's garbage collector — all in-memory frames disappear! By writing each frame to a `cv2.VideoWriter` file as it arrives, the video is 100% complete on disk before the user even clicks Stop.

### After Clicking Stop
```
ctx.video_processor.writer.release() → finalizes the mp4 file →
session_state stores the file path →
"✅ Session recorded (X KB)" message appears →
Download button for the video appears →
User clicks ▶ START ANALYSIS →
The SAME full analysis pipeline (Step 3 above) runs on the recorded video →
Full session summary + PDF report generated
```

---

## 11. THE PDF COACHING REPORT (`src/report_generator.py`)

Uses the `fpdf2` library to generate a professional PDF with:

1. **Header** — Player name, date, session ID
2. **Score Summary Table** — Overall, Balance, Stability, Power, Timing scores
3. **Shot Type & Phase** — What shot was played and at what phase
4. **AI Coach Insights** — Top 4 SHAP-identified weaknesses with specific tips
5. **Injury Flags** — Any dangerous biomechanical patterns detected
6. **Pro Similarity Score** — % similarity to chosen professional

The PDF is saved to `reports/` folder and served as a download button in the app.

---

## 12. THE DATABASE & PLAYER PROGRESS (`src/player_db.py`)

Uses Python's built-in `sqlite3` — no external database server needed.

### Tables
**`sessions` table**: Every analysis run is saved here
```
player_name | timestamp | overall_score | balance_score | stability_score |
power_score | timing_score | shot_type | swing_phase | video_filename |
feedback_json | injury_flags_json
```

### Features Enabled by the Database
- **Player Progress Tab**: Line chart of all scores over time
- **Personal Bests**: Best overall, balance, stability, power, timing across all sessions
- **Shot Distribution**: Bar chart showing which shots the player uses most
- **Radar Chart**: Latest session across all 5 score dimensions

---

## 13. PRO COMPARISON SYSTEM (`src/scorer.py` — `PRO_PROFILES`)

5 professional profiles are hard-coded based on published coaching literature:

| Player | Overall Benchmark | Specialty |
|---|---|---|
| BCCI Reference | 100/100 | Official NCA coaching standard |
| Virat Kohli | 94/100 | Exceptional head stillness + kinetic chain |
| Sachin Tendulkar | 95/100 | Orthodox technique, high elbow, zero head movement |
| Rohit Sharma | 91/100 | Wide stance, elegant timing |
| Steve Smith | 90/100 | Unorthodox but effective, high backlift |

For each profile, 10 key biomechanical features are stored. The similarity score measures the average percentage closeness of the user's session-averaged features to the pro's values.

---

## 14. SETUP GUIDE — RECREATE EVERYTHING FROM SCRATCH

### Prerequisites
- Windows / Mac / Linux
- Python 3.10+ installed
- Git installed
- A webcam or smartphone

### Step 1: Clone the Repository
```bash
git clone https://github.com/sairam-sr8/Sports-Analytics.git
cd Sports-Analytics
```

### Step 2: Create Virtual Environment
```bash
python -m venv cricket_env
# Windows:
cricket_env\Scripts\activate
# Mac/Linux:
source cricket_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Setup
```bash
python run_test.py
```
This runs a diagnostic on all modules and checks model files exist.

### Step 5: Run the App
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

### OPTIONAL: Re-Train the AI From Scratch

#### Step A: Add Training Videos
Place cricket batting videos:
- `videos/good/` → good technique videos (name: `good_drive_001.mp4`)
- `videos/bad/` → poor technique videos (name: `bad_stance_001.mp4`)

#### Step B: Augment the Videos
```bash
python scripts/video_augmenter.py
```
This creates `_aug_flip.mp4` and `_aug_dark.mp4` versions of each video.

#### Step C: Build the Dataset (WARNING: takes 2-6 hours depending on video count)
```bash
python run_dataset_builder.py
```
This creates `data/cricket_biomechanics_dataset.csv`

#### Step D: Train the Models
```bash
python run_ml_trainer.py
```
This creates `models/cricket_model.pkl`, `models/cricket_shap_model.pkl`, `models/feature_meta.json`

---

## 15. GITHUB & DEPLOYMENT

### Repository
**URL**: https://github.com/sairam-sr8/Sports-Analytics  
**Branch**: `main`

### What Is On GitHub (and what isn't)

| File Type | On GitHub? | Reason |
|---|---|---|
| All Python source code | ✅ Yes | Essential for running the app |
| `requirements.txt` | ✅ Yes | Tells Streamlit Cloud what to install |
| `packages.txt` | ✅ Yes | System-level deps for Streamlit Cloud |
| ML models (`*.pkl`) | ✅ Yes | Needed for inference on Streamlit Cloud |
| Training videos (`videos/good/`, `videos/bad/`) | ❌ No | Too large (GBs), excluded via .gitignore |
| Training dataset (`*.csv`) | ❌ No | 34MB, re-generated by dataset_builder |
| SQLite database (`*.db`) | ❌ No | User-specific, not shareable |
| PDF reports | ❌ No | Generated on demand |
| Virtual environment (`cricket_env/`) | ❌ No | Never committed — re-created on each machine |

### Deploying to Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. New app → Repository: `sairam-sr8/Sports-Analytics` → Branch: `main` → Main file: `app.py`
4. Click Deploy

---

## 16. FUTURE ROADMAP

| Feature | Description | Difficulty |
|---|---|---|
| **Bowling Analysis** | Extend feature extractor to measure bowling arm speed, release point | Medium |
| **Wicket-keeping Mode** | Glove tracking, dive angles, reaction time | Medium |
| **Multi-player Sessions** | Analyze multiple batters in one session | Medium |
| **Mobile App** | React Native app using TensorFlow Lite for on-device inference | High |
| **Live WebRTC Cloud** | Ultra-low latency live coaching via dedicated TURN servers | High |
| **Ball Tracking** | Detect the ball and measure bat-ball contact quality | Very High |
| **Team Dashboard** | Coach sees all players' scores in one view | Medium |
| **Video Comparison** | Side-by-side split-screen of user vs pro | Medium |

---

## SUMMARY — THE 3 SENTENCE PITCH

**For a customer**: "Record your batting on your phone, upload the video, and get an instant AI coaching report showing your technique scores, what you need to fix, how you compare to Virat Kohli, and a PDF report your coach can use."

**For a professor**: "We built a markerless biomechanics analysis pipeline using MediaPipe for 33-point skeletal tracking, extracting 22 biomechanical features per frame, and trained an ensemble Random Forest (91.17% accuracy) with SHAP-based Gradient Boosting for explainability, deployed as a Streamlit web application with WebRTC live camera support."

**For an investor**: "We've productized elite cricket coaching analytics — technology previously available only to national-level teams for ₹50+ lakh — into a zero-hardware, smartphone-first SaaS platform that any coaching academy in India can use from day one."

---

*Documentation written for Cricket Biomechanics Analyzer V2*  
*Last updated: May 2026*  
*Author: Sairam SR8 + AI Development Team*
