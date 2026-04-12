"""
========================================================
ml_trainer.py — The AI Judge
========================================================

WHAT THIS FILE DOES (In Plain English):
----------------------------------------
This is Sprint 3: Machine Learning. 

1. Reads our newly created cricket_biomechanics_dataset.csv
2. Mixes up the data and hides 20% of it for testing later.
3. Teaches a "Random Forest" AI using the remaining 80%.
   - The AI looks at Elbow Angles and Knee Bends and learns what makes a Label "1" (Good) vs "0" (Bad).
4. Tests itself on the 20% hidden data to see how accurate it is.
5. Saves its "Brain" as a file (cricket_model.pkl) so we can use it on the website later.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MLTrainer:
    def __init__(self, dataset_path="data/cricket_biomechanics_dataset.csv", model_save_path="models/cricket_model.pkl"):
        self.dataset_path = dataset_path
        self.model_save_path = model_save_path

        # Make sure the models directory exists
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

    def train(self):
        print("\n" + "="*60)
        print("  CRICKET BIOMECHANICS — AI TRAINER (Sprint 3)")
        print("="*60)

        # 1. Load the data
        if not os.path.exists(self.dataset_path):
            print(f"  ❌ ERROR: Dataset not found at {self.dataset_path}")
            return False

        print("  [1/5] Loading Dataset...")
        df = pd.read_csv(self.dataset_path)

        # Clean the data (Drop rows where the AI couldn't calculate angles, denoted by exactly 0.0)
        # We drop 0.0 because if an angle is exactly 0.0, it means the joint wasn't visible on screen
        df_clean = df[(df['right_elbow_angle'] > 0) & (df['right_knee_bend'] > 0)].copy()

        print(f"        Total frames loaded: {len(df)}")
        print(f"        Usable frames after cleaning zero-values: {len(df_clean)}")

        if len(df_clean) < 100:
            print("  ❌ ERROR: Not enough clean data to train the AI.")
            return False

        # 2. Extract Features (X) and Labels (y)
        print("  [2/5] Preparing Features...")
        
        # X is the physical measurements
        feature_columns = ['right_elbow_angle', 'left_elbow_angle', 'right_knee_bend', 'left_knee_bend']
        X = df_clean[feature_columns]
        
        # y is the answer key (1 = Good, 0 = Bad)
        y = df_clean['label']

        # 3. Split into Train and Test Data
        # Hide 20% of the data from the AI so we can test it fairly afterwards
        print("  [3/5] Splitting Data (80% Train, 20% Test)...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Train the Random Forest
        # We use a Random Forest because it is incredibly fast and easily learns rules
        # like "IF right_elbow > 150 AND right_knee < 160 THEN Good"
        print("  [4/5] Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train, y_train)

        # 5. Evaluate the accuracy
        print("  [5/5] Evaluating AI Model...")
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n  =================================================")
        print(f"  🏆 TEST ACCURACY: {accuracy * 100:.2f}%")
        print("  =================================================\n")

        # Feature Importance: Which joint matters most to the AI?
        print("  PHYSICS IMPORTANCE (What the AI cares about):")
        importances = rf_model.feature_importances_
        for name, importance in zip(feature_columns, importances):
            print(f"    - {name:20}: {importance * 100:.1f}%")

        # 6. Save the model 
        joblib.dump(rf_model, self.model_save_path)
        print(f"\n  ✅ AI Brain saved successfully to: {self.model_save_path}")
        print("="*60 + "\n")

        return True
