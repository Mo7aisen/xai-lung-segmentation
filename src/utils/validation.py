#!/usr/bin/env python3
"""
Validation script to verify that models and epochs align with supervisor's specifications
Run this after training to ensure all required models and epochs are available
"""

import os
import pandas as pd
import yaml
from pathlib import Path

# Supervisor's specifications from the meeting
SUPERVISOR_SPECS = {
    "montgomery_full": {
        "underfitting": [5],
        "good_fitting": [75, 105],
        "overfitting": [250]
    },
    "montgomery_half": {
        "underfitting": [10],
        "good_fitting": [115, 140],
        "overfitting": [300]
    },
    "jsrt_full": {
        "underfitting": [5],
        "good_fitting": [35],
        "overfitting": [150]
    },
    "jsrt_half": {
        "underfitting": [5],
        "good_fitting": [70],
        "overfitting": [250]
    }
}


def load_config():
    """Load project configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ config.yaml not found")
        return None


def check_model_exists(run_dir, epoch=None, model_type="snapshot"):
    """Check if a specific model file exists"""
    if model_type == "snapshot" and epoch:
        model_path = os.path.join(run_dir, "snapshots", f"epoch_{epoch}.pth")
    elif model_type == "best":
        model_path = os.path.join(run_dir, "best_model.pth")
    elif model_type == "final":
        model_path = os.path.join(run_dir, "final_model.pth")
    else:
        return False, "Unknown model type"

    exists = os.path.exists(model_path)
    return exists, model_path


def get_training_info(run_dir):
    """Get training information from log file"""
    log_path = os.path.join(run_dir, "training_log.csv")
    if not os.path.exists(log_path):
        return None, None, None

    try:
        df = pd.read_csv(log_path)
        max_epoch = df['epoch'].max()

        # Find best epoch (lowest validation loss)
        best_epoch_idx = df['val_loss'].idxmin()
        best_epoch = df.loc[best_epoch_idx, 'epoch']
        best_val_loss = df.loc[best_epoch_idx, 'val_loss']

        return max_epoch, best_epoch, best_val_loss
    except Exception as e:
        return None, None, None


def validate_supervisor_requirements():
    """Validate that all supervisor requirements can be met"""
    print("🔍 VALIDATING SUPERVISOR'S EPOCH SPECIFICATIONS")
    print("=" * 60)

    config = load_config()
    if not config:
        return False

    output_dir = config.get('output_base_dir', './outputs')
    all_valid = True

    # Map dataset configs to run names
    dataset_to_runs = {
        "montgomery_full": ["unet_montgomery_full_150", "unet_montgomery_full_250"],
        "montgomery_half": ["unet_montgomery_half_150", "unet_montgomery_half_300"],
        "jsrt_full": ["unet_jsrt_full_150"],
        "jsrt_half": ["unet_jsrt_half_150", "unet_jsrt_half_250"]
    }

    for dataset_key, specs in SUPERVISOR_SPECS.items():
        print(f"\n📊 {dataset_key.replace('_', ' ').title()} Dataset:")
        print("-" * 40)

        # Find the best run for this dataset (highest epoch count)
        available_runs = []
        for run_name in dataset_to_runs[dataset_key]:
            run_dir = os.path.join(output_dir, run_name)
            if os.path.exists(run_dir):
                max_epoch, best_epoch, best_val_loss = get_training_info(run_dir)
                if max_epoch:
                    available_runs.append((run_name, max_epoch, best_epoch, best_val_loss))

        if not available_runs:
            print(f"❌ No trained models found for {dataset_key}")
            all_valid = False
            continue

        # Sort by max epoch (use the most complete training)
        available_runs.sort(key=lambda x: x[1], reverse=True)
        primary_run = available_runs[0]
        run_name, max_epoch, best_epoch, best_val_loss = primary_run

        print(f"   Primary run: {run_name}")
        print(f"   Max epoch: {max_epoch}")
        print(f"   Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})")

        run_dir = os.path.join(output_dir, run_name)

        # Check each state requirement
        for state, required_epochs in specs.items():
            print(f"\n   {state.replace('_', ' ').title()}:")

            state_valid = True
            for epoch in required_epochs:
                if state == "underfitting":
                    # Check snapshot
                    exists, path = check_model_exists(run_dir, epoch, "snapshot")
                    status = "✅" if exists else "❌"
                    print(f"     Epoch {epoch}: {status} {path if exists else 'MISSING'}")
                    if not exists:
                        state_valid = False

                elif state == "good_fitting":
                    # Check snapshot first, then best model
                    exists, path = check_model_exists(run_dir, epoch, "snapshot")
                    if not exists:
                        exists, path = check_model_exists(run_dir, model_type="best")
                        path += f" (fallback for epoch {epoch})"

                    status = "✅" if exists else "❌"
                    print(f"     Epoch {epoch}: {status} {path if exists else 'MISSING'}")
                    if not exists:
                        state_valid = False

                elif state == "overfitting":
                    # Check if we have enough epochs
                    if epoch <= max_epoch:
                        # Check final model or snapshot
                        exists, path = check_model_exists(run_dir, model_type="final")
                        if not exists:
                            exists, path = check_model_exists(run_dir, epoch, "snapshot")

                        status = "✅" if exists else "❌"
                        print(f"     Epoch {epoch}: {status} {path if exists else 'MISSING'}")
                        if not exists:
                            state_valid = False
                    else:
                        print(f"     Epoch {epoch}: ❌ Training incomplete (max: {max_epoch})")
                        print(f"       🔧 Solution: Run extended training to {epoch} epochs")
                        state_valid = False

            if not state_valid:
                all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("🎉 ALL SUPERVISOR REQUIREMENTS MET!")
        print("✅ Ready for XAI evaluation with specified epochs")
    else:
        print("⚠️  SOME REQUIREMENTS NOT MET")
        print("💡 Recommendations:")
        print("   1. Complete initial training: ./run_training.sh")
        print("   2. Run extended training: ./run_extended_training.sh")
        print("   3. Check training logs for any failures")

    return all_valid


def show_evaluation_plan():
    """Show the evaluation plan based on supervisor's specs"""
    print("\n🎯 EVALUATION PLAN (Supervisor's Specifications)")
    print("=" * 60)

    for dataset_key, specs in SUPERVISOR_SPECS.items():
        print(f"\n📋 {dataset_key.replace('_', ' ').title()} Dataset:")

        for state, epochs in specs.items():
            epoch_str = ", ".join(map(str, epochs))
            print(f"   • {state.replace('_', ' ').title()}: Epoch(s) {epoch_str}")

    print("\n🚀 Commands to run evaluation:")
    print("   ./run_evaluation.sh")
    print("   ./run_extended_evaluation.sh")

    print("\n📊 Dashboard will show comparative analysis across all states")


def main():
    print("🫁 XAI LUNG SEGMENTATION - SUPERVISOR REQUIREMENTS VALIDATION")
    print("=" * 70)
    print("Based on meeting notes with Dr. Hadházi and Dr. Hullám")
    print()

    is_valid = validate_supervisor_requirements()

    if is_valid:
        show_evaluation_plan()

        print("\n🎨 Next Steps:")
        print("   1. Run evaluation: ./run_evaluation.sh && ./run_extended_evaluation.sh")
        print("   2. Launch dashboard: streamlit run app.py")
        print("   3. Compare models across underfitting → good fitting → overfitting")

    return is_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)