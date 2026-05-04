"""
========================================================
ml_trainer.py — The AI Judge (V2 — 22 Features + SHAP)
========================================================

V2 UPGRADE:
  - Trains on 22 features (up from 4)
  - Trains a separate Gradient Boosting model for SHAP explainability
  - Prints per-feature importance rankings
  - Saves both the model and a feature importance report
  - Evaluates with F1-score and Confusion Matrix

AUTHOR: Cricket Biomechanics Analyzer V2
"""

import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class MLTrainer:
    def __init__(self, dataset_path="data/cricket_biomechanics_dataset.csv", model_dir="models"):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = os.path.join(base_dir, dataset_path)
        self.model_dir = os.path.join(base_dir, model_dir)
        self.model_save_path = os.path.join(self.model_dir, "cricket_model.pkl")
        self.shap_model_path = os.path.join(self.model_dir, "cricket_shap_model.pkl")
        
        os.makedirs(self.model_dir, exist_ok=True)

    # ── All 22 features ───────────────────────────────────────
    FEATURE_COLUMNS = [
        'right_elbow_angle',        'left_elbow_angle',
        'right_shoulder_elevation', 'left_shoulder_elevation',
        'shoulder_rotation_angle',  'wrist_height_diff',
        'bat_swing_arc',
        'right_knee_bend',          'left_knee_bend',
        'right_hip_angle',          'left_hip_angle',
        'hip_rotation_angle',       'trunk_lean_angle',
        'spine_angle',
        'center_of_mass_x',         'center_of_mass_y',
        'feet_width_ratio',         'weight_distribution',
        'head_stability_x',         'head_stability_y',
        'follow_through_angle',     'backlift_height',
    ]

    def train(self):
        print("\n" + "="*60)
        print("  CRICKET BIOMECHANICS — AI TRAINER (V2 — 22 Features)")
        print("="*60)

        # 1. Load Dataset
        if not os.path.exists(self.dataset_path):
            print(f"  [ERROR] Dataset not found at {self.dataset_path}")
            print("     Run: python run_dataset_builder.py first.")
            return False

        print("  [1/6] Loading Dataset...")
        df = pd.read_csv(self.dataset_path)
        print(f"        Total rows loaded: {len(df)}")

        # 2. Feature selection — use only columns available in this dataset
        available_features = [f for f in self.FEATURE_COLUMNS if f in df.columns]
        print(f"  [2/6] Features available: {len(available_features)}/{len(self.FEATURE_COLUMNS)}")

        # Clean zero-value rows (0.0 = landmark not visible, invalid data)
        df_clean = df[
            (df['right_elbow_angle'] > 0) &
            (df['right_knee_bend']   > 0)
        ].copy()
        print(f"        Usable frames after cleaning: {len(df_clean)}")

        if len(df_clean) < 100:
            print("  ❌ ERROR: Not enough clean data to train. Need 100+ rows.")
            return False

        # 3. Prepare X and y
        print("  [3/6] Preparing Features & Labels...")
        X = df_clean[available_features].fillna(0.0)
        y = df_clean['label']

        # Class distribution
        class_counts = y.value_counts()
        print(f"        Class distribution — Good: {class_counts.get(1, 0)}, "
              f"Bad: {class_counts.get(0, 0)}")

        # 4. Train/Test split
        print("  [4/6] Splitting Data (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. Train primary Random Forest (used in live inference)
        print("  [5/6] Training Random Forest Classifier (22 features)...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)

        # Also train a Gradient Boosting model (better for SHAP)
        print("        Training Gradient Boosting model (for SHAP explainability)...")
        gb_model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        gb_model.fit(X_train, y_train)

        # 6. Evaluate
        print("  [6/6] Evaluating Models...")
        rf_pred   = rf_model.predict(X_test)
        gb_pred   = gb_model.predict(X_test)

        rf_acc    = accuracy_score(y_test, rf_pred)
        gb_acc    = accuracy_score(y_test, gb_pred)
        rf_f1     = f1_score(y_test, rf_pred, average='weighted')

        # Cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')

        print("\n  " + "="*50)
        print(f"  [RF]  Random Forest -- Test Accuracy : {rf_acc*100:.2f}%")
        print(f"  [RF]  Random Forest -- F1 Score      : {rf_f1*100:.2f}%")
        print(f"  [RF]  Random Forest -- CV Score      : {cv_scores.mean()*100:.2f}% +/-{cv_scores.std()*100:.2f}%")
        print(f"  [GB]  Gradient Boost -- Test Accuracy: {gb_acc*100:.2f}%")
        print("  " + "="*50 + "\n")

        print("  CONFUSION MATRIX (Random Forest):")
        cm = confusion_matrix(y_test, rf_pred)
        print(f"    True Positives  (Good predicted Good): {cm[1][1]}")
        print(f"    True Negatives  (Bad predicted Bad):   {cm[0][0]}")
        print(f"    False Positives (Bad predicted Good):  {cm[0][1]}")
        print(f"    False Negatives (Good predicted Bad):  {cm[1][0]}")

        print("\n  FEATURE IMPORTANCE (What the AI cares about most):")
        importances = rf_model.feature_importances_
        feat_imp = sorted(
            zip(available_features, importances),
            key=lambda x: x[1], reverse=True
        )
        for fname, imp in feat_imp:
            bar = "█" * int(imp * 100)
            bar = "#" * int(imp * 100)
            print(f"    {fname:<30} {imp*100:5.1f}%  {bar}")

        # Save both models
        joblib.dump(rf_model, self.model_save_path)
        joblib.dump(gb_model, self.shap_model_path)

        print(f"\n  [SAVED] Random Forest -> {self.model_save_path}")
        print(f"  [SAVED] SHAP GB model -> {self.shap_model_path}")

        # Save feature names used (important for inference alignment)
        feature_meta = {
            'feature_columns': available_features,
            'rf_accuracy':     round(rf_acc, 4),
            'rf_f1':           round(float(rf_f1), 4),
            'cv_mean':         round(float(cv_scores.mean()), 4),
        }
        import json
        with open("models/feature_meta.json", "w") as f:
            json.dump(feature_meta, f, indent=2)
        print("  [SAVED] Feature metadata -> models/feature_meta.json")
        print("="*60 + "\n")

        return True
