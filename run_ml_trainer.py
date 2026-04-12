import os
import sys

# Add 'src' to path so it can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from ml_trainer import MLTrainer

def main():
    try:
        trainer = MLTrainer()
        trainer.train()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Python script failed: {e}")

if __name__ == "__main__":
    main()
