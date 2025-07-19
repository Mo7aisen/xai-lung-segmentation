#!/usr/bin/env python3
"""
Script to organize model checkpoints based on supervisor's notes
This will copy specific epoch models to organized folders for easy access
"""

import os
import shutil
import yaml
from pathlib import Path


def organize_checkpoints(config_path='config.yaml'):
    """
    Organize model checkpoints according to the meeting notes
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    output_base_dir = config['output_base_dir']

    # Define model epochs based on supervisor's notes
    model_epochs = {
        'montgomery_full': {
            'underfitting': [5],
            'best_fitting': [75, 105],
            'overfitting': [250]  # Need to train this
        },
        'montgomery_half': {
            'underfitting': [10],
            'best_fitting': [115, 140],
            'overfitting': [300]  # Need to train this
        },
        'jsrt_full': {
            'underfitting': [5],
            'best_fitting': [35],
            'overfitting': [150]
        },
        'jsrt_half': {
            'underfitting': [5],
            'best_fitting': [70],
            'overfitting': [250]  # Need to train this
        }
    }

    # Map to actual experiment names
    experiment_mapping = {
        'montgomery_full': ['unet_montgomery_full_150', 'unet_montgomery_full_50'],
        'montgomery_half': ['unet_montgomery_half_150', 'unet_montgomery_half_50'],
        'jsrt_full': ['unet_jsrt_full_150', 'unet_jsrt_full_50'],
        'jsrt_half': ['unet_jsrt_half_150', 'unet_jsrt_half_50']
    }

    # Create organized directory structure
    organized_dir = os.path.join(output_base_dir, 'organized_models')
    os.makedirs(organized_dir, exist_ok=True)

    copy_log = []
    missing_log = []

    for model_key, epochs_dict in model_epochs.items():
        model_dir = os.path.join(organized_dir, model_key)

        for category, epochs in epochs_dict.items():
            category_dir = os.path.join(model_dir, category)
            os.makedirs(category_dir, exist_ok=True)

            for epoch in epochs:
                # Find the source model file
                found = False
                for exp_name in experiment_mapping[model_key]:
                    source_path = os.path.join(output_base_dir, exp_name, 'snapshots', f'epoch_{epoch}.pth')

                    if os.path.exists(source_path):
                        dest_path = os.path.join(category_dir, f'epoch_{epoch}.pth')
                        shutil.copy2(source_path, dest_path)
                        copy_log.append(f"Copied: {model_key}/{category}/epoch_{epoch}.pth")
                        found = True
                        break

                if not found:
                    missing_log.append(f"Missing: {model_key}/{category}/epoch_{epoch}.pth")

    # Save organization report
    report_path = os.path.join(organized_dir, 'organization_report.txt')
    with open(report_path, 'w') as f:
        f.write("Model Checkpoint Organization Report\n")
        f.write("=====================================\n\n")

        f.write("Successfully Copied:\n")
        for item in copy_log:
            f.write(f"  ✓ {item}\n")

        f.write("\nMissing Checkpoints (need training):\n")
        for item in missing_log:
            f.write(f"  ✗ {item}\n")

        f.write("\nNext Steps:\n")
        f.write("1. Train Montgomery full dataset to 250 epochs\n")
        f.write("2. Train Montgomery half dataset to 300 epochs\n")
        f.write("3. Train JSRT half dataset to 250 epochs\n")

    print(f"Organization complete! Report saved to: {report_path}")
    return copy_log, missing_log


if __name__ == "__main__":
    organize_checkpoints()