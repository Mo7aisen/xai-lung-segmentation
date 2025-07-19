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

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor
from utils import plot_and_save_loss_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(run_name, config):
    """
    Main training loop, now driven by a configuration dictionary.
    Includes early stopping and saving of the best model.
    """
    # 1. Get parameters from config
    params = config['experiments'][run_name]
    dataset_name = params['dataset']
    epochs = params['epochs']
    data_fraction = params['data_fraction']

    output_base_dir = config['output_base_dir']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    save_checkpoint_freq = config['save_freq']
    num_workers = config.get('num_workers', 4)  # Use get for backwards compatibility
    patience = config.get('early_stopping_patience', 15)

    # 2. Setup Output Paths
    run_dir = os.path.join(output_base_dir, run_name)
    snapshots_dir = os.path.join(run_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "training_log.csv")
    plot_file = os.path.join(run_dir, "loss_curve.png")
    final_model_path = os.path.join(run_dir, "final_model.pth")
    best_model_path = os.path.join(run_dir, "best_model.pth")  # Path for the best model

    start_time = time.time()
    print(f"--- Starting Training for '{run_name}' ---")
    print(f"All outputs will be saved in: {run_dir}")

    # 3. Create Dataset and Dataloaders
    try:
        dataset = LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())
    except Exception as e:
        print(f"FATAL: Error creating dataset: {e}");
        return

    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    if data_fraction < 1.0:
        train_set_size = int(len(train_set) * data_fraction)
        train_set = Subset(train_set, range(train_set_size))
        n_train = len(train_set)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=num_workers,
                            pin_memory=True)

    # 4. Initialize Model, Optimizer, and Loss Function
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    training_log = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)
                loss = criterion(model(images), true_masks)
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()
                train_loss += loss.item() * images.size(0)
                pbar.update(images.size(0))

        avg_train_loss = train_loss / n_train if n_train > 0 else 0

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)
                val_loss += criterion(model(images), true_masks).item() * images.size(0)
        avg_val_loss = val_loss / n_val if n_val > 0 else 0

        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
        training_log.append({'epoch': epoch + 1, 'train_loss': avg_train_loss, 'val_loss': avg_val_loss})

        if (epoch + 1) % save_checkpoint_freq == 0:
            torch.save(model.state_dict(), os.path.join(snapshots_dir, f'epoch_{epoch + 1}.pth'))

        # --- Best Model and Early Stopping Logic ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved to {best_model_path} (Val Loss: {best_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  -> Val loss did not improve for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"--- Early stopping triggered after {patience} epochs with no improvement. ---")
            break

    # 6. Finalize
    torch.save(model.state_dict(), final_model_path)
    pd.DataFrame(training_log).to_csv(log_file, index=False)
    plot_and_save_loss_curve(log_file, plot_file)
    print(f"--- Training Finished in {(time.time() - start_time) / 60:.2f} minutes ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True, help="The name of the experiment run from config.yaml")
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    train_model(args.run_name, config)
