# 🏏 Markerless Cricket Biomechanics Analyzer

An advanced, markerless AI engine that evaluates cricket batting technique using standard smartphone videos. It replaces expensive motion-capture hardware by leveraging **Google MediaPipe** and **Machine Learning** to provide instant, pro-level biomechanical feedback, shot classification, and Explainable AI (SHAP) coaching tips.

## ✨ Features
* **Markerless 3D Motion Capture**: Maps 33 skeletal landmarks frame-by-frame via MediaPipe.
* **22-Feature Biomechanical Engine**: Extracts exact joint angles, knee bends, and swing arcs.
* **ML Technique Scoring**: A Random Forest model evaluates the quality of the shot (Balance, Stability, Power, Timing).
* **ML Shot Classification**: Automatically detects shot types (Cover Drive, Pull, Sweep, etc.).
* **Explainable AI Coach**: Uses SHAP to provide personalized coaching tips (e.g., "Your front knee is too stiff").
* **Pro Benchmarks**: Compare your metrics against elite international cricketers.
* **Automated PDF Reports**: Export beautiful coaching reports for sharing.
* **Real-time Live Camera**: Analyze your swing live via WebRTC.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501` to use the Dashboard!

## 📂 Project Architecture
* `app.py`: Streamlit Dashboard (UI).
* `src/pose_detector.py`: MediaPipe Digital X-Ray engine.
* `src/feature_extractor.py`: Mathematics and Trigonometry for 22 features.
* `src/dataset_builder.py`: Generates the ML dataset from videos.
* `src/ml_trainer.py`: Trains the scoring model & SHAP explainer.
* `src/shot_ml_trainer.py`: Trains the Shot Classification model.
* `src/report_generator.py`: PDF Coaching Report generator.
* `scripts/video_augmenter.py`: Data augmentation pipeline.

## 🧠 Training Your Own Model
If you add new videos to `videos/good/` and `videos/bad/`, you can re-train the models:
1. Augment Data: `python scripts/video_augmenter.py`
2. Build Dataset: `python run_dataset_builder.py`
3. Train Technique Model: `python run_ml_trainer.py`
4. Train Shot Model: `python src/shot_ml_trainer.py`

## 🛠 Tech Stack
* **Python 3.11+**
* **Computer Vision**: OpenCV, MediaPipe Tasks API
* **Data Science**: Pandas, NumPy
* **Machine Learning**: Scikit-learn, SHAP
* **Web UI**: Streamlit, Streamlit-WebRTC
* **Exports**: FPDF2
