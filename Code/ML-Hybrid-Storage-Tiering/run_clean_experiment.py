"""
Phase 21 — Clean Experiment Reproducibility Script.

This script orchestrates the end-to-end pipeline from a clean state:
1. Resets the simulated storage environment.
2. Initializes storage from metadata.
3. Generates features (if not existing).
4. Trains the ML model (to verify reproducibility).
5. Runs the evaluation (which covers predictions, payback, decision, feedback, and baselines).
"""

import sys
import shutil
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))
import config

from migration.storage_manager import establish_initial_tiers
from features.feature_extractor import build_ml_dataset
from ml.data_preparation import prepare_ml_data
from ml.train_model import train_and_evaluate
from evaluation.run_evaluation import run_full_evaluation

def reset_storage_environment():
    print("--- 1. Resetting Storage Environment ---")
    if config.STORAGE_DIR.exists():
        shutil.rmtree(config.STORAGE_DIR)
        print("Deleted existing storage directories.")
    (config.STORAGE_DIR / "HDD").mkdir(parents=True, exist_ok=True)
    (config.STORAGE_DIR / "SSD").mkdir(parents=True, exist_ok=True)
    print("Created fresh storage directories.")

def initialize_storage():
    print("\n--- 2. Initializing Simulated Storage ---")
    establish_initial_tiers()
    print("Storage initialized from metadata.")

def ensure_features_and_splits():
    print("\n--- 3. Generating Features and Temporal Splits ---")
    if not config.PROCESSED_DATASET_PATH.exists():
        build_ml_dataset(config.TRACE_DATASET_PATH, config.METADATA_DATASET_PATH, config.PROCESSED_DATASET_PATH)
        print("Generated ML features.")
    else:
        print("ML features already exist.")
        
    if not config.TRAIN_DATASET_PATH.exists():
        prepare_ml_data()
        print("Created train/val/test splits.")
    else:
        print("Temporal splits already exist.")

def retrain_model():
    print("\n--- 4. Training ML Model (Reproducibility Check) ---")
    train_and_evaluate()
    print("Model trained and saved successfully.")

def run_evaluation_pipeline():
    print("\n--- 5. Running Comprehensive Evaluation Pipeline ---")
    run_full_evaluation()
    print("Comprehensive evaluation complete. Results saved in evaluation/results/.")

def main():
    print("============================================================")
    print("PHASE 21: CLEAN EXPERIMENT REPRODUCIBILITY TEST")
    print("============================================================")
    
    try:
        reset_storage_environment()
        initialize_storage()
        ensure_features_and_splits()
        retrain_model()
        run_evaluation_pipeline()
        
        print("\n============================================================")
        print("CLEAN EXPERIMENT SUCCESSFUL")
        print("============================================================")
    except Exception as e:
        print(f"\n[ERROR] Clean experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
