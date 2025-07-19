import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
import csv
from captum.attr import IntegratedGradients
import yaml

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor
from utils import get_data_splits, dice_score, iou_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """Enable dropout layers for uncertainty estimation"""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def generate_ig_map(model, input_tensor, device):
    """Generate Integrated Gradients attribution map"""
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()

    def model_forward_wrapper(inp):
        return model(inp).sum().unsqueeze(0)

    ig = IntegratedGradients(model_forward_wrapper)
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(input_tensor, baselines=baseline, n_steps=25)
    return attributions.squeeze().cpu().detach().numpy()


def generate_uncertainty_map(model, input_tensor, device, n_samples=25):
    """Generate uncertainty map using Monte Carlo dropout"""
    enable_dropout(model)
    input_tensor = input_tensor.to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            predictions.append(torch.sigmoid(model(input_tensor)).cpu())

    return torch.var(torch.stack(predictions), dim=0).squeeze().numpy()


def get_supervisor_epoch_mapping():
    """
    Get epoch mapping based on supervisor's meeting notes
    """
    return {
        # Montgomery Full Dataset
        "unet_montgomery_full_150": {
            "underfitting": [5],
            "good_fitting": [75, 105],
            "overfitting": [150]  # Will use final_model.pth
        },
        "unet_montgomery_full_50": {
            "underfitting": [5],
            "good_fitting": [50],  # Best available in 50-epoch run
            "overfitting": [50]  # Final model, likely overfitting
        },
        "unet_montgomery_full_250": {
            "underfitting": [5],
            "good_fitting": [75, 105],
            "overfitting": [250]
        },

        # Montgomery Half Dataset
        "unet_montgomery_half_150": {
            "underfitting": [10],
            "good_fitting": [115, 140],
            "overfitting": [150]
        },
        "unet_montgomery_half_50": {
            "underfitting": [10],
            "good_fitting": [50],
            "overfitting": [50]
        },
        "unet_montgomery_half_300": {
            "underfitting": [10],
            "good_fitting": [115, 140],
            "overfitting": [300]
        },

        # JSRT Full Dataset
        "unet_jsrt_full_150": {
            "underfitting": [5],
            "good_fitting": [35],
            "overfitting": [150]
        },
        "unet_jsrt_full_50": {
            "underfitting": [5],
            "good_fitting": [35],  # Best available in 50-epoch run
            "overfitting": [50]
        },

        # JSRT Half Dataset
        "unet_jsrt_half_150": {
            "underfitting": [5],
            "good_fitting": [70],
            "overfitting": [150]
        },
        "unet_jsrt_half_50": {
            "underfitting": [5],
            "good_fitting": [50],
            "overfitting": [50]
        },
        "unet_jsrt_half_250": {
            "underfitting": [5],
            "good_fitting": [70],
            "overfitting": [250]
        }
    }


def get_model_path_for_state(run_dir, run_name, state):
    """Get the appropriate model path based on supervisor's specifications"""

    supervisor_mapping = get_supervisor_epoch_mapping()

    if run_name in supervisor_mapping:
        epochs_for_state = supervisor_mapping[run_name][state]

        # For good_fitting, try the specified epochs in order
        if state == "good_fitting":
            for epoch in epochs_for_state:
                # First try snapshots
                snapshot_path = os.path.join(run_dir, "snapshots", f"epoch_{epoch}.pth")
                if os.path.exists(snapshot_path):
                    return snapshot_path, epoch

            # Fallback to best_model.pth
            best_model_path = os.path.join(run_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                return best_model_path, "best"

        # For underfitting, use the specified epoch
        elif state == "underfitting":
            epoch = epochs_for_state[0]  # Take the first (and usually only) epoch
            snapshot_path = os.path.join(run_dir, "snapshots", f"epoch_{epoch}.pth")
            if os.path.exists(snapshot_path):
                return snapshot_path, epoch

        # For overfitting, use final model or highest epoch
        elif state == "overfitting":
            target_epoch = epochs_for_state[0]

            # If target epoch matches a common final epoch, use final_model.pth
            if target_epoch in [150, 250, 300]:
                final_model_path = os.path.join(run_dir, "final_model.pth")
                if os.path.exists(final_model_path):
                    return final_model_path, target_epoch

            # Otherwise try snapshot
            snapshot_path = os.path.join(run_dir, "snapshots", f"epoch_{target_epoch}.pth")
            if os.path.exists(snapshot_path):
                return snapshot_path, target_epoch

    # Fallback to default behavior
    print(f"Warning: Using fallback model selection for {run_name}/{state}")
    fallback_files = ["best_model.pth", "final_model.pth"]
    for model_file in fallback_files:
        model_path = os.path.join(run_dir, model_file)
        if os.path.exists(model_path):
            return model_path, "fallback"

    return None, None


def evaluate_model(run_name, state, split, config):
    """Main evaluation function with supervisor's epoch specifications"""
    print(f"--- Starting Evaluation: {run_name} | {state} | {split} ---")

    # 1. Get parameters from config
    if run_name not in config['experiments']:
        # Handle extended experiments that might not be in config
        print(f"Run '{run_name}' not found in config. Attempting to infer parameters...")
        parts = run_name.split('_')
        if len(parts) >= 4:
            dataset_name = parts[1]  # montgomery or jsrt
            data_fraction = 1.0 if parts[2] == 'full' else 0.5
        else:
            print(f"Error: Cannot parse run name '{run_name}'")
            return
    else:
        params = config['experiments'][run_name]
        dataset_name = params['dataset']

    run_dir = os.path.join(config['output_base_dir'], run_name)

    # 2. Get model path based on supervisor's specifications
    model_path, epoch_info = get_model_path_for_state(run_dir, run_name, state)

    if not model_path:
        print(f"No suitable model found for {run_name}/{state}")
        print("Available files:")
        try:
            for file in os.listdir(run_dir):
                if file.endswith('.pth'):
                    print(f"  - {file}")
            snapshots_dir = os.path.join(run_dir, 'snapshots')
            if os.path.exists(snapshots_dir):
                print("  Snapshots:")
                for file in sorted(os.listdir(snapshots_dir)):
                    if file.endswith('.pth'):
                        print(f"    - snapshots/{file}")
        except:
            pass
        return

    print(f"Using model: {os.path.relpath(model_path, run_dir)} (epoch {epoch_info})")

    # Show supervisor's mapping for this run
    supervisor_mapping = get_supervisor_epoch_mapping()
    if run_name in supervisor_mapping:
        print(f"Supervisor's mapping for {run_name}:")
        for s, epochs in supervisor_mapping[run_name].items():
            print(f"  {s}: epochs {epochs}")

    # 3. Load Model
    try:
        model = UNet(n_channels=1, n_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✓ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Prepare Dataset and Dataloader
    try:
        full_dataset = LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())
        train_set, val_set, test_set = get_data_splits(full_dataset)

        split_map = {"training": train_set, "validation": val_set, "test": test_set}
        data_to_evaluate = split_map[split]
        data_loader = DataLoader(data_to_evaluate, batch_size=1, shuffle=False)
        print(f"✓ Dataset loaded: {len(data_to_evaluate)} samples in {split} split")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 5. Setup Output Directory
    split_output_dir = os.path.join(run_dir, "evaluation", state, split)
    maps_dir = os.path.join(split_output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    # 6. Initialize Results Logging
    csv_file_path = os.path.join(split_output_dir, "_results_log.csv")
    with open(csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_name", "dice_score", "iou_score"])

        # 7. Evaluation Loop
        results = []
        model.eval()

        for i, batch in enumerate(tqdm(data_loader, desc=f"Evaluating {dataset_name} '{split}' set")):
            try:
                image_tensor, mask_tensor = batch['image'].to(DEVICE), batch['mask']
                image_name = os.path.basename(full_dataset.image_files[data_to_evaluate.indices[i]])

                # Generate predictions
                with torch.no_grad():
                    pred_probs = torch.sigmoid(model(image_tensor))

                # Calculate metrics
                dice = dice_score((pred_probs > 0.5).cpu(), mask_tensor)
                iou = iou_score((pred_probs > 0.5).cpu(), mask_tensor)
                csv_writer.writerow([image_name, f"{dice:.6f}", f"{iou:.6f}"])

                # Generate XAI maps
                try:
                    ig_map = generate_ig_map(model, image_tensor, DEVICE)
                    uncertainty_map = generate_uncertainty_map(model, image_tensor, DEVICE)
                except Exception as e:
                    print(f"Warning: XAI generation failed for {image_name}: {e}")
                    # Create dummy maps to avoid breaking the pipeline
                    ig_map = np.zeros_like(pred_probs.cpu().squeeze().numpy())
                    uncertainty_map = np.zeros_like(pred_probs.cpu().squeeze().numpy())

                # Save results
                npz_path = os.path.join(maps_dir, f"{image_name}.npz")
                np.savez_compressed(
                    npz_path,
                    ig_map=ig_map,
                    uncertainty_map=uncertainty_map,
                    prediction=pred_probs.cpu().squeeze().numpy(),
                    ground_truth=mask_tensor.squeeze().numpy()
                )

                results.append({
                    "image_name": image_name,
                    "dice_score": dice,
                    "iou_score": iou,
                    "xai_results_path": npz_path
                })

                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    # 8. Save Evaluation Summary
    if results:
        summary = {
            "model_path": model_path,
            "epoch_used": epoch_info,
            "supervisor_specification": supervisor_mapping.get(run_name, {}).get(state, []),
            "dataset_name": dataset_name,
            "split": split,
            "state": state,
            "num_samples": len(results),
            "average_dice_score": np.mean([r['dice_score'] for r in results]),
            "average_iou_score": np.mean([r['iou_score'] for r in results]),
            "per_sample_results": results
        }

        summary_path = os.path.join(split_output_dir, "_evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"✓ Evaluation completed successfully!")
        print(f"  - Model epoch: {epoch_info}")
        print(f"  - Processed {len(results)} samples")
        print(f"  - Average Dice Score: {summary['average_dice_score']:.4f}")
        print(f"  - Average IoU Score: {summary['average_iou_score']:.4f}")
        print(f"  - Results saved to: {summary_path}")
    else:
        print("✗ No results generated. Check for errors above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models with supervisor's epoch specifications")
    parser.add_argument('--run_name', type=str, required=True,
                        help="Name of the experiment run")
    parser.add_argument('--state', type=str, required=True,
                        choices=['underfitting', 'good_fitting', 'overfitting'],
                        help="Model state to evaluate")
    parser.add_argument('--split', type=str, required=True,
                        choices=['training', 'validation', 'test'],
                        help="Data split to evaluate")
    args = parser.parse_args()

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please ensure it exists in the current directory.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        exit(1)

    evaluate_model(args.run_name, args.state, args.split, config)