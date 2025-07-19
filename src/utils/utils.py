import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from torch.utils.data import random_split
import numpy as np


def plot_and_save_loss_curve(log_file_path, output_path):
    """
    Plot training and validation loss curves from CSV log file

    Args:
        log_file_path (str): Path to the training log CSV file
        output_path (str): Path where to save the plot
    """
    try:
        log_df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"Warning: Log file not found: {log_file_path}")
        return

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    plt.plot(log_df['epoch'], log_df['train_loss'], 'b-o', label='Training Loss', markersize=4)
    plt.plot(log_df['epoch'], log_df['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    plt.title(f'Training & Validation Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if not log_df['val_loss'].empty and log_df['val_loss'].notna().any():
        best_epoch = log_df.loc[log_df['val_loss'].idxmin()]
        plt.axvline(x=best_epoch['epoch'], color='g', linestyle='--',
                    label=f"Best Model (Epoch {int(best_epoch['epoch'])})")

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_data_splits(full_dataset, train_percent=0.7, val_percent=0.15):
    """
    Split dataset into train, validation, and test sets

    Args:
        full_dataset: The complete dataset
        train_percent (float): Percentage for training set
        val_percent (float): Percentage for validation set

    Returns:
        tuple: (train_set, val_set, test_set)
    """
    n_total = len(full_dataset)
    n_train = int(n_total * train_percent)
    n_val = int(n_total * val_percent)
    n_test = n_total - n_train - n_val
    return random_split(full_dataset, [n_train, n_val, n_test],
                        generator=torch.Generator().manual_seed(99))


def dice_score(pred, target, smooth=1e-6):
    """
    Calculate Dice score between prediction and target

    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        float: Dice score
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target.float()).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union) score

    Args:
        pred: Predicted mask
        target: Ground truth mask
        smooth (float): Smoothing factor to avoid division by zero

    Returns:
        float: IoU score
    """
    pred = (pred > 0.5).float()
    intersection = (pred * target.float()).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def find_best_epoch(log_csv_path):
    """
    Find the epoch with the best (lowest) validation loss

    Args:
        log_csv_path (str): Path to training log CSV file

    Returns:
        int or None: Best epoch number, None if file not found or invalid
    """
    try:
        log_df = pd.read_csv(log_csv_path)
        if 'val_loss' not in log_df.columns or log_df['val_loss'].isna().all():
            return None
        return int(log_df.loc[log_df['val_loss'].idxmin()]['epoch'])
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Warning: Could not find best epoch from {log_csv_path}: {e}")
        return None


def calculate_cumulative_histogram(attribution_map, center_point):
    """
    Calculate cumulative histogram of attribution weights by distance from center point

    Args:
        attribution_map (np.array): 2D attribution map
        center_point (tuple): (y, x) coordinates of center point

    Returns:
        tuple: (normalized_hist, fig) - cumulative histogram and matplotlib figure
    """
    y, x = np.mgrid[0:attribution_map.shape[0], 0:attribution_map.shape[1]]
    distances = np.sqrt((y - center_point[0]) ** 2 + (x - center_point[1]) ** 2)
    weights = np.abs(attribution_map)

    # Create DataFrame for easier processing
    df = pd.DataFrame({
        'distance': distances.flatten(),
        'weight': weights.flatten()
    })

    # Create distance bins
    max_dist = np.ceil(df['distance'].max())
    bins = np.arange(0, max_dist + 2)
    df['distance_bin'] = pd.cut(df['distance'], bins=bins, right=False,
                                labels=bins[:-1]).astype(float)

    # Calculate weighted histogram
    weighted_hist = df.groupby('distance_bin')['weight'].sum()
    cumulative_hist = weighted_hist.cumsum()
    normalized_hist = cumulative_hist / cumulative_hist.max() if cumulative_hist.max() > 0 else cumulative_hist

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x=normalized_hist.index, y=normalized_hist.values, ax=ax, marker='o')
    ax.set_title("Individual Cumulative Weighted Attribution")
    ax.set_xlabel("Distance (pixels)")
    ax.set_ylabel("Normalized Score")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_dist)

    return normalized_hist, fig


def create_directory_structure(base_dir):
    """
    Create the standard directory structure for the project

    Args:
        base_dir (str): Base directory path
    """
    directories = [
        'outputs',
        'outputs/logs',
        'outputs/organized_models',
        'outputs/extended_training_logs'
    ]

    for directory in directories:
        full_path = os.path.join(base_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Created directory: {full_path}")


def validate_config(config):
    """
    Validate the configuration file structure

    Args:
        config (dict): Configuration dictionary

    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ['output_base_dir', 'datasets', 'experiments']

    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in config")
            return False

    # Validate datasets
    for dataset_name, dataset_config in config['datasets'].items():
        required_dataset_keys = ['path', 'images', 'masks']
        for key in required_dataset_keys:
            if key not in dataset_config:
                print(f"Error: Missing key '{key}' in dataset '{dataset_name}'")
                return False

        # Check if dataset path exists
        if not os.path.exists(dataset_config['path']):
            print(f"Warning: Dataset path does not exist: {dataset_config['path']}")

    print("✓ Configuration validation passed")
    return True


def get_experiment_info(run_name):
    """
    Extract experiment information from run name

    Args:
        run_name (str): Name of the experiment run

    Returns:
        dict: Dictionary with experiment information
    """
    parts = run_name.replace("unet_", "").split('_')

    if len(parts) >= 3:
        return {
            "dataset": parts[0],
            "data_size": parts[1],
            "epochs": int(parts[2]) if parts[2].isdigit() else parts[2]
        }
    else:
        return {
            "dataset": "unknown",
            "data_size": "unknown",
            "epochs": "unknown"
        }


def check_gpu_availability():
    """
    Check if GPU is available and print information

    Returns:
        bool: True if GPU available, False otherwise
    """
    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024 ** 3} GB")
        return True
    else:
        print("⚠ No GPU available, using CPU")
        return False


def format_file_size(size_bytes):
    """
    Format file size in human readable format

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log2(size_bytes) / 10))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def get_model_info(model_path):
    """
    Get information about a model file

    Args:
        model_path (str): Path to model file

    Returns:
        dict: Model information
    """
    if not os.path.exists(model_path):
        return {"exists": False}

    stat = os.stat(model_path)
    return {
        "exists": True,
        "size": format_file_size(stat.st_size),
        "modified": pd.Timestamp.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    }