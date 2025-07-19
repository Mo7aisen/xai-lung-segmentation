import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import time
import yaml
import shutil

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor
from utils import plot_and_save_loss_curve, check_gpu_availability

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_base_experiment(dataset, data_size, config):
    """
    Find the base experiment to continue training from

    Args:
        dataset (str): Dataset name (montgomery/jsrt)
        data_size (str): Data size (full/half)
        config (dict): Configuration dictionary

    Returns:
        tuple: (base_run_name, base_epochs) or (None, None) if not found
    """
    # Look for existing experiments that match the dataset and data size
    possible_epochs = [150, 50]  # Common epoch numbers from initial training

    for epochs in possible_epochs:
        run_name = f"unet_{dataset}_{data_size}_{epochs}"
        if run_name in config['experiments']:
            run_dir = os.path.join(config['output_base_dir'], run_name)
            if os.path.exists(os.path.join(run_dir, 'final_model.pth')):
                return run_name, epochs

    return None, None


def continue_training(dataset, data_size, target_epochs, config):
    """
    Continue training from existing checkpoint to reach target epochs

    Args:
        dataset (str): Dataset name (montgomery/jsrt)
        data_size (str): Data size (full/half)
        target_epochs (int): Target number of epochs
        config (dict): Configuration dictionary
    """
    print(f"=== Extended Training: {dataset}_{data_size} to {target_epochs} epochs ===")

    # Find base experiment to continue from
    base_run_name, base_epochs = find_base_experiment(dataset, data_size, config)

    if not base_run_name:
        print(f"Error: No base experiment found for {dataset}_{data_size}")
        print("Please run initial training first with ./run_training.sh")
        return False

    print(f"Continuing from: {base_run_name} ({base_epochs} epochs)")

    # Setup paths
    output_base_dir = config['output_base_dir']
    source_dir = os.path.join(output_base_dir, base_run_name)

    # Create target run name and directory
    target_run_name = f"unet_{dataset}_{data_size}_{target_epochs}"
    target_dir = os.path.join(output_base_dir, target_run_name)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Target directory: {target_dir}")

    # Copy existing files if target is empty
    if not os.path.exists(os.path.join(target_dir, 'training_log.csv')):
        print("Copying existing training results...")
        items_to_copy = ['training_log.csv', 'loss_curve.png', 'best_model.pth']

        for item in items_to_copy:
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)

            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"  ✓ Copied {item}")

        # Copy snapshots directory
        source_snapshots = os.path.join(source_dir, 'snapshots')
        target_snapshots = os.path.join(target_dir, 'snapshots')
        if os.path.exists(source_snapshots):
            shutil.copytree(source_snapshots, target_snapshots, dirs_exist_ok=True)
            print(f"  ✓ Copied snapshots directory")

    # Load existing training log
    log_file = os.path.join(target_dir, "training_log.csv")
    if os.path.exists(log_file):
        training_log_df = pd.read_csv(log_file)
        training_log = training_log_df.to_dict('records')
        start_epoch = len(training_log)
        print(f"Resuming from epoch {start_epoch + 1}")
    else:
        training_log = []
        start_epoch = 0
        print("Starting from scratch (no existing log found)")

    # Check if training is already complete
    if start_epoch >= target_epochs:
        print(f"Training already complete! Current epochs: {start_epoch}, Target: {target_epochs}")
        return True

    # Load model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)

    # Try to load from the best model first, then final model
    model_candidates = [
        os.path.join(target_dir, "best_model.pth"),
        os.path.join(source_dir, "best_model.pth"),
        os.path.join(target_dir, "final_model.pth"),
        os.path.join(source_dir, "final_model.pth")
    ]

    model_loaded = False
    for model_path in model_candidates:
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                print(f"✓ Loaded model from: {os.path.relpath(model_path)}")
                model_loaded = True
                break
            except Exception as e:
                print(f"✗ Failed to load {model_path}: {e}")
                continue

    if not model_loaded:
        print("Warning: No model checkpoint found, starting from random weights")

    # Create dataset and dataloaders
    data_fraction = 1.0 if data_size == 'full' else 0.5

    try:
        dataset = LungDataset(dataset_name=dataset, config=config, transform=ResizeAndToTensor())
        print(f"✓ Loaded dataset: {len(dataset)} total images")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False

    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(42))

    if data_fraction < 1.0:
        train_set_size = int(len(train_set) * data_fraction)
        train_set = Subset(train_set, range(train_set_size))
        n_train = len(train_set)
        print(f"Using {data_fraction * 100:.0f}% of training data: {n_train} samples")

    # Setup data loaders
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                            batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Setup training components
    learning_rate = config.get('learning_rate', 0.0001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Extended training settings
    save_checkpoint_freq = 10  # Save every 10 epochs for extended training
    snapshots_dir = os.path.join(target_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    # Best model tracking
    best_val_loss = float('inf')
    if training_log:
        # Find current best validation loss
        val_losses = [entry['val_loss'] for entry in training_log if 'val_loss' in entry]
        if val_losses:
            best_val_loss = min(val_losses)
            print(f"Current best validation loss: {best_val_loss:.6f}")

    print(f"\nStarting extended training from epoch {start_epoch + 1} to {target_epochs}")
    print(f"Device: {DEVICE}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, target_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{target_epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)

                # Forward pass
                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar.update(images.size(0))

        avg_train_loss = train_loss / n_train if n_train > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)
                masks_pred = model(images)
                val_loss += criterion(masks_pred, true_masks).item() * images.size(0)

        avg_val_loss = val_loss / n_val if n_val > 0 else 0

        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Update training log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(target_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  → New best model saved (Val Loss: {best_val_loss:.6f})")

        # Save checkpoint
        if (epoch + 1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(snapshots_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Checkpoint saved: epoch_{epoch + 1}.pth")

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model and update logs
    final_model_path = os.path.join(target_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Save updated training log
    pd.DataFrame(training_log).to_csv(log_file, index=False)

    # Update loss curve plot
    plot_file = os.path.join(target_dir, "loss_curve.png")
    plot_and_save_loss_curve(log_file, plot_file)

    training_time = (time.time() - start_time) / 60
    print(f"\n✓ Extended training completed in {training_time:.2f} minutes!")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Best validation loss: {best_val_loss:.6f}")
    print(f"  - Training log: {log_file}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Continue training to demonstrate overfitting")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['montgomery_full', 'montgomery_half', 'jsrt_half'],
                        help='Dataset configuration to train')
    parser.add_argument('--target_epochs', type=int, required=True,
                        help='Target number of epochs to reach')
    args = parser.parse_args()

    # Check GPU
    check_gpu_availability()

    # Load configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        return False

    # Parse dataset argument
    parts = args.dataset.split('_')
    if len(parts) != 2:
        print(
            f"Error: Invalid dataset format '{args.dataset}'. Expected format: 'dataset_size' (e.g., 'montgomery_full')")
        return False

    dataset, data_size = parts

    # Validate arguments
    if dataset not in ['montgomery', 'jsrt']:
        print(f"Error: Unknown dataset '{dataset}'. Must be 'montgomery' or 'jsrt'")
        return False

    if data_size not in ['full', 'half']:
        print(f"Error: Unknown data size '{data_size}'. Must be 'full' or 'half'")
        return False

    if args.target_epochs < 50:
        print(f"Error: Target epochs ({args.target_epochs}) should be at least 50")
        return False

    # Start extended training
    success = continue_training(dataset, data_size, args.target_epochs, config)

    if success:
        print(f"\n🎉 Training completed successfully!")
        print(f"Next step: Run evaluation with:")
        print(
            f"  python3 evaluate.py --run_name unet_{dataset}_{data_size}_{args.target_epochs} --state overfitting --split test")
    else:
        print(f"\n❌ Training failed. Check the logs above for errors.")
        return False

    return True


if __name__ == "__main__":
    main()