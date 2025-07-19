import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.patches import Circle
import matplotlib.patches as patches
from scipy import ndimage
import cv2
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(
    page_title="🫁 XAI Lung Segmentation Analysis - Enhanced",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Constants and Configuration ---
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    OUTPUT_DIR = CONFIG['output_base_dir']
    DATASETS = CONFIG['datasets']
except Exception as e:
    st.error(f"Warning: Could not load config.yaml: {e}")
    # Environment variable fallback with default
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', "./outputs")
    DATASETS = {}

# Data path configuration from environment
DATA_PATH = os.getenv('DATA_PATH', '/home/mohaisen_mohammed/Datasets')

# --- Enhanced XAI Analysis Functions ---

def calculate_median_distance(attr_z, x, y, min_abs_attr=None, max_abs_attr=None):
    """
    Compute a weighted median distance using absolute attribution scores as weights after thresholding.
    
    Args:
        attr_z: Attribution map array
        x: X coordinate of center point
        y: Y coordinate of center point
        min_abs_attr: Minimum absolute attribution threshold
        max_abs_attr: Maximum absolute attribution threshold
        
    Returns:
        str: Formatted median distance or 'N/A'
    """
    if min_abs_attr is None:
        min_abs_attr = np.percentile(np.abs(attr_z), 5)
    if max_abs_attr is None:
        max_abs_attr = np.percentile(np.abs(attr_z), 95)
        
    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)
    
    if np.any(selected_mask):
        selected_indices = np.where(selected_mask)
        distances = np.sqrt((selected_indices[0] - y) ** 2 + (selected_indices[1] - x) ** 2)
        weights = np.abs(attr_z[selected_mask])
        
        if len(distances) > 0:
            sorted_idx = np.argsort(distances)
            sorted_distances = distances[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1]
            
            if total_weight > 0:
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0]
                if len(median_idx) > 0:
                    return f"{sorted_distances[median_idx[0]]:.2f}"
    
    return 'N/A'


def calculate_weighted_median_distance_vectorized(attr_map, center_x, center_y, min_abs_attr=None, max_abs_attr=None):
    """Vectorized version for better performance"""
    attr_z = attr_map.copy()

    if min_abs_attr is None:
        min_abs_attr = np.percentile(np.abs(attr_z), 5)
    if max_abs_attr is None:
        max_abs_attr = np.percentile(np.abs(attr_z), 95)

    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)

    if np.any(selected_mask):
        # Vectorized distance calculation
        y_coords, x_coords = np.mgrid[0:attr_z.shape[0], 0:attr_z.shape[1]]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        
        # Apply mask
        valid_distances = distances[selected_mask]
        weights = np.abs(attr_z[selected_mask])

        if len(valid_distances) > 0:
            sorted_idx = np.argsort(valid_distances)
            sorted_distances = valid_distances[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1] if len(cumulative_weights) > 0 else 0

            if total_weight > 0:
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0]
                if len(median_idx) > 0:
                    median_distance = sorted_distances[median_idx[0]]
                    return median_distance, sorted_distances, cumulative_weights, total_weight

    return None, None, None, None


def calculate_enhanced_cumulative_histogram_vectorized(attr_map, center_x, center_y):
    """Vectorized version of cumulative histogram calculation"""
    # Vectorized coordinate grid
    y_coords, x_coords = np.mgrid[0:attr_map.shape[0], 0:attr_map.shape[1]]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    weights = np.abs(attr_map)

    # Flatten arrays
    distances_flat = distances.flatten()
    weights_flat = weights.flatten()

    # Filter out very small weights
    threshold = np.percentile(weights_flat, 5)
    significant_mask = weights_flat > threshold
    distances_clean = distances_flat[significant_mask]
    weights_clean = weights_flat[significant_mask]

    if len(distances_clean) == 0:
        return None, None

    # Sort by distance
    sorted_idx = np.argsort(distances_clean)
    sorted_distances = distances_clean[sorted_idx]
    sorted_weights = weights_clean[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)

    # Normalize
    if cumulative_weights[-1] > 0:
        normalized_cumulative = cumulative_weights / cumulative_weights[-1]
    else:
        normalized_cumulative = cumulative_weights

    # Create distance bins
    max_dist = int(np.ceil(sorted_distances.max()))
    distance_bins = np.linspace(0, max_dist, min(max_dist + 1, 200))
    binned_cumulative = np.interp(distance_bins, sorted_distances, normalized_cumulative)

    return distance_bins, binned_cumulative


# --- Enhanced Helper Functions ---

def get_model_info(run_name):
    """Extract comprehensive model information from run name"""
    parts = run_name.replace("unet_", "").split('_')
    if len(parts) >= 3:
        return {
            "model_type": "U-Net",
            "dataset": parts[0].upper(),
            "data_size": parts[1].title(),
            "epochs": str(parts[2]),
            "display_name": f"{parts[0].upper()} {parts[1].title()} {parts[2]}ep"
        }
    return {
        "model_type": "U-Net",
        "dataset": "Unknown",
        "data_size": "Unknown", 
        "epochs": "Unknown",
        "display_name": "Unknown Model"
    }


def safe_file_operation(func, *args, **kwargs):
    """Safely execute file operations with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        st.error(f"File operation failed: {e}")
        return None


def get_available_model_states(run_name):
    """Get all available model states with detailed information"""
    eval_dir = os.path.join(OUTPUT_DIR, run_name, "evaluation")
    if not os.path.exists(eval_dir):
        return []

    states = []
    state_mapping = {
        'underfitting': {'display': 'Underfitting', 'description': 'Early training state'},
        'good_fitting': {'display': 'Good Fitting', 'description': 'Optimal model state'},
        'overfitting': {'display': 'Overfitting', 'description': 'Over-trained state'}
    }
    
    try:
        for state in ['underfitting', 'good_fitting', 'overfitting']:
            state_dir = os.path.join(eval_dir, state)
            if os.path.exists(state_dir):
                info = state_mapping.get(state, {'display': state.title(), 'description': 'Model state'})
                states.append({
                    'name': state,
                    'display': info['display'],
                    'description': info['description']
                })
    except Exception as e:
        st.error(f"Error scanning model states: {e}")

    return states


# --- Enhanced Data Loading Functions ---

@st.cache_data
def load_model_data(run_name, state, split):
    """Enhanced data loading with comprehensive error handling"""
    try:
        with st.spinner(f"Loading {run_name} data..."):
            summary_path = os.path.join(OUTPUT_DIR, run_name, "evaluation", state, split, "_evaluation_summary.json")
            
            if not os.path.exists(summary_path):
                return None
                
            with open(summary_path, 'r') as f:
                data = json.load(f)
                
            # Add metadata
            data['_metadata'] = {
                'run_name': run_name,
                'state': state,
                'split': split,
                'loaded_at': pd.Timestamp.now().isoformat()
            }
            
            return data
            
    except Exception as e:
        st.error(f"Failed to load model data: {e}")
        return None


@st.cache_data
def load_npz_with_validation(npz_path):
    """Load NPZ data with validation and error handling"""
    if not os.path.exists(npz_path):
        st.error(f"NPZ file not found: {npz_path}")
        return None

    try:
        with np.load(npz_path) as data:
            result = {key: data[key] for key in data}
            
            # Validate required fields
            required_fields = ['ig_map']
            for field in required_fields:
                if field not in result:
                    st.warning(f"Missing field in NPZ: {field}")
                    
            return result
            
    except Exception as e:
        st.error(f"Error loading NPZ file: {e}")
        return None


def get_dataset_image_path(image_name, dataset_name):
    """Get image path with environment variable support"""
    # Use environment variables first, then config, then fallback
    if dataset_name.lower() in DATASETS:
        dataset_config = DATASETS[dataset_name.lower()]
        base_path = os.path.join(dataset_config['path'], dataset_config['images'])
        return os.path.join(base_path, image_name)
    
    # Environment variable fallback
    if dataset_name.lower() == 'montgomery':
        montgomery_path = os.getenv('MONTGOMERY_PATH', os.path.join(DATA_PATH, 'MontgomeryDataset/CXR_png'))
        return os.path.join(montgomery_path, image_name)
    else:
        jsrt_path = os.getenv('JSRT_PATH', os.path.join(DATA_PATH, 'JSRT/images'))
        return os.path.join(jsrt_path, image_name)


def load_original_image_enhanced(image_path):
    """Enhanced image loading with multiple format support"""
    try:
        if not os.path.exists(image_path):
            st.error(f"Image not found: {image_path}")
            return None, None
            
        # Load image with PIL
        original_image = Image.open(image_path).convert("L")
        original_array = np.array(original_image)

        # Get dimensions
        original_height, original_width = original_array.shape

        return original_array, (original_width, original_height)
        
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None, None


def create_combined_overlay_image(original_img, ground_truth, prediction, threshold=0.5, alpha_gt=0.3, alpha_pred=0.3):
    """Create combined overlay of original image with ground truth and prediction masks"""
    try:
        # Ensure all images are the same size
        if original_img.shape != ground_truth.shape:
            # Resize ground truth to match original
            ground_truth = cv2.resize(ground_truth.astype(np.float32), 
                                    (original_img.shape[1], original_img.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
        
        if original_img.shape != prediction.shape:
            # Resize prediction to match original  
            prediction = cv2.resize(prediction.astype(np.float32), 
                                  (original_img.shape[1], original_img.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
        
        # Apply threshold to prediction
        prediction_binary = (prediction > threshold).astype(np.float32)
        
        # Normalize original image to 0-1 range
        original_norm = original_img.astype(np.float32) / np.max(original_img) if np.max(original_img) > 0 else original_img.astype(np.float32)
        
        # Create RGB version of original image
        combined = np.stack([original_norm, original_norm, original_norm], axis=-1)
        
        # Add ground truth in red channel
        combined[:, :, 0] = np.where(ground_truth > 0.5, 
                                   combined[:, :, 0] * (1 - alpha_gt) + alpha_gt, 
                                   combined[:, :, 0])
        
        # Add prediction in blue channel
        combined[:, :, 2] = np.where(prediction_binary > 0.5, 
                                   combined[:, :, 2] * (1 - alpha_pred) + alpha_pred, 
                                   combined[:, :, 2])
        
        # Overlap regions will appear purple
        overlap = (ground_truth > 0.5) & (prediction_binary > 0.5)
        combined[overlap, 1] = combined[overlap, 1] * (1 - alpha_pred/2) + alpha_pred/2  # Add some green for purple effect
        
        return np.clip(combined, 0, 1)
        
    except Exception as e:
        st.error(f"Error creating overlay image: {e}")
        return None


def create_enhanced_attribution_map(attr_map, center_x, center_y, title, show_grid_centers=True):
    """Enhanced attribution map with grid centers and improved visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Apply Gaussian smoothing
    smoothed_attr = ndimage.gaussian_filter(attr_map, sigma=1.0)

    # Enhanced color normalization
    vmax = np.percentile(np.abs(smoothed_attr), 98)
    vmin = -vmax

    # Main attribution map
    im1 = ax1.imshow(smoothed_attr, cmap='RdBu_r', interpolation='bilinear',
                     vmin=vmin, vmax=vmax, alpha=0.9)

    # Professional crosshair
    ax1.axhline(y=center_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
    ax1.axvline(x=center_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

    # Target marker
    circle = Circle((center_x, center_y), radius=8, color='#00FF00', fill=False, linewidth=3)
    ax1.add_patch(circle)
    ax1.plot(center_x, center_y, 'o', color='#00FF00', markersize=8,
             markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

    # Add grid centers if requested
    if show_grid_centers:
        grid_size = 16  # Adjust based on attribution map resolution
        for i in range(grid_size//2, attr_map.shape[0], grid_size):
            for j in range(grid_size//2, attr_map.shape[1], grid_size):
                ax1.plot(j, i, '+', color='yellow', markersize=4, markeredgewidth=1)

    ax1.set_title(f"{title} - Smoothed View\n🎯 Analysis Point: ({center_x}, {center_y})",
                  fontsize=16, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel("X Coordinate", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Y Coordinate", fontsize=14, fontweight='bold')

    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Attribution Score\n🔴 Positive | 🔵 Negative',
                    rotation=270, labelpad=20, fontsize=14, fontweight='bold')

    # Raw attribution map
    im2 = ax2.imshow(attr_map, cmap='RdBu_r', interpolation='nearest',
                     vmin=vmin, vmax=vmax, alpha=0.9)

    ax2.axhline(y=center_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
    ax2.axvline(x=center_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

    circle2 = Circle((center_x, center_y), radius=8, color='#00FF00', fill=False, linewidth=3)
    ax2.add_patch(circle2)
    ax2.plot(center_x, center_y, 'o', color='#00FF00', markersize=8,
             markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

    # Add grid centers to raw map too
    if show_grid_centers:
        for i in range(grid_size//2, attr_map.shape[0], grid_size):
            for j in range(grid_size//2, attr_map.shape[1], grid_size):
                ax2.plot(j, i, '+', color='yellow', markersize=4, markeredgewidth=1)

    ax2.set_title(f"{title} - Raw Data\n🎯 Analysis Point: ({center_x}, {center_y})",
                  fontsize=16, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel("X Coordinate", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Y Coordinate", fontsize=14, fontweight='bold')

    # Colorbar for raw map
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('Attribution Score\n🔴 Positive | 🔵 Negative',
                    rotation=270, labelpad=20, fontsize=14, fontweight='bold')

    # Enhanced statistics
    pos_attr = np.sum(attr_map[attr_map > 0])
    neg_attr = np.sum(attr_map[attr_map < 0])
    max_pos = np.max(attr_map)
    min_neg = np.min(attr_map)
    center_value = attr_map[center_y, center_x] if 0 <= center_y < attr_map.shape[0] and 0 <= center_x < attr_map.shape[1] else 0

    # Calculate additional metrics
    mean_attr = np.mean(attr_map)
    std_attr = np.std(attr_map)
    
    stats_text = (f"Center Value: {center_value:.6f}\n"
                  f"Mean: {mean_attr:.6f}\n"
                  f"Std: {std_attr:.6f}\n"
                  f"Max Positive: {max_pos:.6f}\n"
                  f"Min Negative: {min_neg:.6f}\n"
                  f"Total Positive: {pos_attr:.3f}\n"
                  f"Total Negative: {neg_attr:.3f}")

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    return fig


def create_interactive_image_selector(image_array, overlay_image, current_x, current_y, original_size, column_id):
    """Create enhanced interactive image selector with overlay support"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Original image
    ax1.imshow(image_array, cmap='gray', extent=[0, original_size[0], original_size[1], 0])
    
    # Add crosshair and marker to original
    ax1.axhline(y=current_y, color='#FF4444', linestyle='-', alpha=0.8, linewidth=3)
    ax1.axvline(x=current_x, color='#FF4444', linestyle='-', alpha=0.8, linewidth=3)
    
    circle1 = Circle((current_x, current_y), radius=15, color='#FF4444', fill=False, linewidth=4)
    ax1.add_patch(circle1)
    ax1.plot(current_x, current_y, 'o', color='#FF4444', markersize=8,
            markerfacecolor='white', markeredgecolor='#FF4444', markeredgewidth=3)

    ax1.set_title(f"Original X-ray Image\n🎯 Selected Point: ({current_x}, {current_y})",
                 fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel("X Coordinate (pixels)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Y Coordinate (pixels)", fontsize=12, fontweight='bold')
    ax1.set_xlim(0, original_size[0])
    ax1.set_ylim(original_size[1], 0)

    # Overlay image
    if overlay_image is not None:
        ax2.imshow(overlay_image, extent=[0, original_size[0], original_size[1], 0])
        
        # Add crosshair and marker to overlay
        ax2.axhline(y=current_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
        ax2.axvline(x=current_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
        
        circle2 = Circle((current_x, current_y), radius=15, color='#00FF00', fill=False, linewidth=4)
        ax2.add_patch(circle2)
        ax2.plot(current_x, current_y, 'o', color='#00FF00', markersize=8,
                markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=3)

        ax2.set_title(f"Combined Overlay (🔴 Ground Truth | 🔵 Prediction)\n🎯 Selected Point: ({current_x}, {current_y})",
                     fontsize=14, fontweight='bold', color='#2C3E50')
        ax2.set_xlabel("X Coordinate (pixels)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Y Coordinate (pixels)", fontsize=12, fontweight='bold')
        ax2.set_xlim(0, original_size[0])
        ax2.set_ylim(original_size[1], 0)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', alpha=0.7, linewidth=4, label='Ground Truth'),
            Line2D([0], [0], color='blue', alpha=0.7, linewidth=4, label='Prediction'),
            Line2D([0], [0], color='purple', alpha=0.7, linewidth=4, label='Overlap')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'Overlay image not available', 
                transform=ax2.transAxes, fontsize=16, ha='center', va='center')
        ax2.set_title("Combined Overlay (Not Available)", fontsize=14, fontweight='bold', color='#8B0000')

    plt.tight_layout()
    return fig


def export_analysis_results(analysis_data, export_path):
    """Export analysis results to files with comprehensive error handling"""
    try:
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        exported_files = []
        
        # Export attribution map
        if 'attribution_map' in analysis_data:
            attr_path = os.path.join(export_path, 'attribution_map.npy')
            np.save(attr_path, analysis_data['attribution_map'])
            exported_files.append('attribution_map.npy')
        
        # Export metrics as CSV
        if 'metrics' in analysis_data:
            metrics_path = os.path.join(export_path, 'metrics.csv')
            metrics_df = pd.DataFrame([analysis_data['metrics']])
            metrics_df.to_csv(metrics_path, index=False)
            exported_files.append('metrics.csv')
        
        # Export cumulative histogram data
        if 'histogram_data' in analysis_data:
            hist_path = os.path.join(export_path, 'cumulative_histogram.csv')
            hist_data = analysis_data['histogram_data']
            if hist_data['distance_bins'] is not None and hist_data['cumulative_values'] is not None:
                hist_df = pd.DataFrame({
                    'distance_bins': hist_data['distance_bins'],
                    'cumulative_values': hist_data['cumulative_values']
                })
                hist_df.to_csv(hist_path, index=False)
                exported_files.append('cumulative_histogram.csv')
        
        # Export comparison data if available
        if 'comparison_data' in analysis_data:
            comp_path = os.path.join(export_path, 'comparison_data.json')
            with open(comp_path, 'w') as f:
                json.dump(analysis_data['comparison_data'], f, indent=2, default=str)
            exported_files.append('comparison_data.json')
        
        # Create a summary file
        summary_path = os.path.join(export_path, 'export_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Export Summary\n")
            f.write(f"==============\n")
            f.write(f"Export Date: {pd.Timestamp.now().isoformat()}\n")
            f.write(f"Export Path: {export_path}\n")
            f.write(f"Exported Files: {', '.join(exported_files)}\n")
        exported_files.append('export_summary.txt')
        
        return True, exported_files
        
    except Exception as e:
        st.error(f"Export failed: {e}")
        return False, []


def display_enhanced_column(column_id, available_runs):
    """Enhanced column display with all new features"""
    st.markdown(f"### 🔬 Enhanced Analysis Column {column_id}")

    # Model selection with enhanced display
    selected_run = st.selectbox(
        f"Select Model Run",
        [""] + available_runs,
        key=f"run_{column_id}",
        format_func=lambda name: get_model_info(name)['display_name'] if name else "Select a model run...",
        help="Choose from available trained models"
    )

    if not selected_run:
        st.info("Please select a model run to begin analysis.")
        return None

    # Enhanced state selection
    available_states = get_available_model_states(selected_run)
    if not available_states:
        st.warning(f"No evaluation results found for {selected_run}")
        return None

    selected_state_info = st.selectbox(
        f"Select Model State",
        available_states,
        key=f"state_{column_id}",
        format_func=lambda x: f"{x['display']} - {x['description']}",
        help="Select the training state to analyze"
    )
    selected_state = selected_state_info['name']

    # Split selection
    selected_split = st.selectbox(
        f"Select Data Split",
        ["test", "validation", "training"],
        key=f"split_{column_id}",
        help="Choose which dataset split to analyze"
    )

    # Load model data with progress indication
    run_data = load_model_data(selected_run, selected_state, selected_split)
    if not run_data:
        st.error(f"Could not load data for {selected_run}/{selected_state}/{selected_split}")
        return None

    # Image selection with enhanced metrics display
    results = run_data.get("per_sample_results", [])
    if not results:
        st.warning("No sample results found.")
        return None

    # Sort by Dice score for easier selection
    sorted_results = sorted(results, key=lambda x: x.get("dice_score", 0), reverse=True)

    selected_image_data = st.selectbox(
        f"Select Image (Sorted by Dice Score)",
        sorted_results,
        key=f"image_{column_id}",
        format_func=lambda x: f"{x['image_name']} | 🎯 Dice: {x['dice_score']:.4f} | 📐 IoU: {x['iou_score']:.4f}",
        help="Images sorted by Dice score (highest first)"
    )

    # Display enhanced metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Dice Score", f"{selected_image_data['dice_score']:.4f}")
    with col2:
        st.metric("📐 IoU Score", f"{selected_image_data['iou_score']:.4f}")
    with col3:
        model_info = get_model_info(selected_run)
        st.metric("🏗️ Model Type", model_info['model_type'])

    # Load XAI data
    npz_data = load_npz_with_validation(selected_image_data["xai_results_path"])
    if not npz_data:
        st.error(f"Failed to load XAI data for {selected_image_data['image_name']}")
        return None

    # Load images
    model_info = get_model_info(selected_run)
    original_image_path = get_dataset_image_path(selected_image_data['image_name'], model_info['dataset'])
    original_array, original_size = load_original_image_enhanced(original_image_path)
    
    if original_array is None:
        st.error("Could not load original image")
        return None

    # Segmentation threshold slider
    st.markdown("### 🎚️ Segmentation Threshold")
    segmentation_threshold = st.slider(
        "Adjust prediction threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key=f"threshold_{column_id}",
        help="Adjust the threshold for binary segmentation"
    )

    # Create overlay image
    ground_truth = npz_data.get('ground_truth')
    prediction = npz_data.get('predicted_mask', npz_data.get('prediction'))
    
    overlay_image = None
    if ground_truth is not None and prediction is not None:
        overlay_image = create_combined_overlay_image(
            original_array, ground_truth, prediction, 
            threshold=segmentation_threshold
        )

    # Coordinate selection with enhanced interface
    st.markdown("### 🎯 Interactive Coordinate Selection")
    
    # Initialize coordinates
    if f"orig_x_{column_id}" not in st.session_state:
        st.session_state[f"orig_x_{column_id}"] = original_size[0] // 2
    if f"orig_y_{column_id}" not in st.session_state:
        st.session_state[f"orig_y_{column_id}"] = original_size[1] // 2

    # Coordinate controls (removed center and random buttons)
    coord_col1, coord_col2 = st.columns([1, 1])

    with coord_col1:
        orig_x = st.slider(
            "🔄 X Coordinate",
            0, original_size[0] - 1,
            st.session_state[f"orig_x_{column_id}"],
            key=f"orig_x_slider_{column_id}",
            help=f"Select X coordinate (0 to {original_size[0] - 1})"
        )

    with coord_col2:
        orig_y = st.slider(
            "🔄 Y Coordinate", 
            0, original_size[1] - 1,
            st.session_state[f"orig_y_{column_id}"],
            key=f"orig_y_slider_{column_id}",
            help=f"Select Y coordinate (0 to {original_size[1] - 1})"
        )

    # Update session state
    st.session_state[f"orig_x_{column_id}"] = orig_x
    st.session_state[f"orig_y_{column_id}"] = orig_y

    # Map coordinates to analysis space (256x256)
    mapped_x = int((orig_x / original_size[0]) * 256)
    mapped_y = int((orig_y / original_size[1]) * 256)
    mapped_x = max(0, min(mapped_x, 255))
    mapped_y = max(0, min(mapped_y, 255))

    # Display coordinate information
    st.info(f"📍 **Original:** ({orig_x}, {orig_y}) | **Analysis:** ({mapped_x}, {mapped_y}) | **Threshold:** {segmentation_threshold:.2f}")

    # Analysis button
    analyze_button = st.button(
        "🔬 Analyze Selected Point",
        key=f"analyze_{column_id}",
        help="Run Integrated Gradients analysis",
        type="primary"
    )

    # Display enhanced image selector
    try:
        fig = create_interactive_image_selector(original_array, overlay_image, orig_x, orig_y, original_size, column_id)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.error(f"Error displaying image selector: {e}")

    # Analysis results
    if analyze_button or f"analysis_done_{column_id}" in st.session_state:
        st.session_state[f"analysis_done_{column_id}"] = True

        st.markdown("### 🧠 Integrated Gradients Analysis")
        
        try:
            if 'ig_map' in npz_data:
                # Display enhanced attribution map
                fig = create_enhanced_attribution_map(
                    npz_data['ig_map'],
                    mapped_x, mapped_y,
                    "Integrated Gradients Attribution",
                    show_grid_centers=True
                )
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Calculate enhanced metrics
                median_dist, sorted_distances, cumulative_weights, total_weight = calculate_weighted_median_distance_vectorized(
                    npz_data['ig_map'], mapped_x, mapped_y
                )

                # Display metrics with proper formatting
                st.markdown("### 📊 Analysis Metrics")
                
                # Use wider columns for better metric display
                metric_col1, metric_col2 = st.columns(2)
                
                with metric_col1:
                    center_value = npz_data['ig_map'][mapped_y, mapped_x] if 0 <= mapped_y < npz_data['ig_map'].shape[0] and 0 <= mapped_x < npz_data['ig_map'].shape[1] else 0
                    st.metric(
                        label="🎯 Center Attribution",
                        value=f"{center_value:.6f}",
                        help="Attribution value at selected pixel"
                    )
                    
                    max_attribution = np.max(np.abs(npz_data['ig_map']))
                    st.metric(
                        label="📈 Max Attribution",
                        value=f"{max_attribution:.6f}",
                        help="Maximum absolute attribution in the map"
                    )

                with metric_col2:
                    if median_dist is not None:
                        st.metric(
                            label="📏 Weighted Median Distance",
                            value=f"{median_dist:.2f} px",
                            help="Weighted median distance from selected point"
                        )
                    else:
                        st.metric(
                            label="📏 Weighted Median Distance", 
                            value="N/A",
                            help="Could not calculate median distance"
                        )
                    
                    # Updated median distance calculation
                    median_dist_str = calculate_median_distance(npz_data['ig_map'], mapped_x, mapped_y)
                    st.metric(
                        label="📏 Median Distance (Enhanced)",
                        value=median_dist_str,
                        help="Enhanced median distance calculation"
                    )

                # Enhanced cumulative histogram
                distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram_vectorized(
                    npz_data['ig_map'], mapped_x, mapped_y
                )

                if distance_bins is not None:
                    # Stack histograms vertically for better visibility
                    fig = create_professional_cumulative_plot(
                        distance_bins, cumulative_values, (orig_x, orig_y), median_dist
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add professional attribution distribution histogram below
                    hist_fig = create_professional_attribution_histogram(npz_data['ig_map'])
                    st.plotly_chart(hist_fig, use_container_width=True)

                # Export functionality
                col_export1, col_export2 = st.columns([3, 1])
                with col_export2:
                    if st.button("📤 Export Results", key=f"export_{column_id}"):
                        export_data = {
                            'attribution_map': npz_data['ig_map'],
                            'metrics': {
                                'center_attribution': float(center_value),
                                'median_distance': float(median_dist) if median_dist is not None else None,
                                'max_attribution': float(max_attribution),
                                'original_coords': (orig_x, orig_y),
                                'analysis_coords': (mapped_x, mapped_y),
                                'dice_score': selected_image_data['dice_score'],
                                'threshold': segmentation_threshold
                            },
                            'histogram_data': {
                                'distance_bins': distance_bins.tolist() if distance_bins is not None else None,
                                'cumulative_values': cumulative_values.tolist() if cumulative_values is not None else None
                            }
                        }
                        
                        export_path = f"./exports/column_{column_id}_{selected_run}_{selected_state}"
                        success, exported_files = export_analysis_results(export_data, export_path)
                        if success:
                            st.success(f"✅ Exported {len(exported_files)} files to {export_path}")
                            with st.expander("📁 Exported Files"):
                                for file in exported_files:
                                    st.write(f"• {file}")
                        else:
                            st.error("❌ Export failed")

            else:
                st.warning("Integrated Gradients data not available")

        except Exception as e:
            st.error(f"Error in analysis: {e}")

        # Uncertainty analysis with enhanced visualization
        st.markdown("### 🔬 Uncertainty Analysis")
        try:
            if 'uncertainty_map' in npz_data:
                fig, ax = plt.subplots(figsize=(12, 10))

                # Ensure uncertainty map size matches attribution map
                uncertainty_map = npz_data['uncertainty_map']
                if uncertainty_map.shape != npz_data['ig_map'].shape:
                    uncertainty_map = cv2.resize(uncertainty_map, 
                                               (npz_data['ig_map'].shape[1], npz_data['ig_map'].shape[0]),
                                               interpolation=cv2.INTER_LINEAR)

                im = ax.imshow(uncertainty_map, cmap='plasma', interpolation='bilinear', alpha=0.9)

                # Add crosshair
                ax.axhline(y=mapped_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
                ax.axvline(x=mapped_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

                circle = Circle((mapped_x, mapped_y), radius=8, color='#00FF00', fill=False, linewidth=3)
                ax.add_patch(circle)
                ax.plot(mapped_x, mapped_y, 'o', color='#00FF00', markersize=8,
                        markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

                ax.set_title(f"Model Prediction Uncertainty\n🎯 Analysis Point: ({mapped_x}, {mapped_y})",
                             fontsize=16, fontweight='bold', color='#2C3E50')

                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25)
                cbar.set_label('Uncertainty Level\n🟡 High | 🟣 Low',
                               rotation=270, labelpad=25, fontsize=14, fontweight='bold')

                # Enhanced uncertainty statistics
                uncertainty_at_point = uncertainty_map[mapped_y, mapped_x] if 0 <= mapped_y < uncertainty_map.shape[0] and 0 <= mapped_x < uncertainty_map.shape[1] else 0
                mean_uncertainty = np.mean(uncertainty_map)
                max_uncertainty = np.max(uncertainty_map)
                std_uncertainty = np.std(uncertainty_map)

                stats_text = (f"Point Uncertainty: {uncertainty_at_point:.4f}\n"
                              f"Mean Uncertainty: {mean_uncertainty:.4f}\n"
                              f"Std Uncertainty: {std_uncertainty:.4f}\n"
                              f"Max Uncertainty: {max_uncertainty:.4f}")

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            else:
                st.warning("Uncertainty map not available")

        except Exception as e:
            st.error(f"Error displaying uncertainty analysis: {e}")

    # Return enhanced data for comparison
    try:
        if f"analysis_done_{column_id}" in st.session_state and 'ig_map' in npz_data:
            distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram_vectorized(
                npz_data['ig_map'], mapped_x, mapped_y
            )

            return {
                "run_name": f"Col {column_id}: {model_info['display_name']}",
                "hist_data": {"distance_bins": distance_bins, "cumulative_values": cumulative_values},
                "state": selected_state,
                "state_display": selected_state_info['display'],
                "center_point": (orig_x, orig_y),
                "analysis_point": (mapped_x, mapped_y),
                "dice_score": selected_image_data['dice_score'],
                "original_size": original_size,
                "threshold": segmentation_threshold,
                "model_info": model_info
            }
    except Exception as e:
        st.error(f"Error preparing return data: {e}")

    return None


def create_professional_cumulative_plot(distance_bins, cumulative_values, center_point, median_dist=None):
    """Rectangular cumulative plot with full title visibility"""
    fig = go.Figure()

    # Main line with enhanced styling
    fig.add_trace(go.Scatter(
        x=distance_bins,
        y=cumulative_values,
        mode='lines',
        name='Cumulative',
        line=dict(color='#2E86AB', width=3, shape='spline', smoothing=1.3),
        fill='tonexty',
        fillcolor='rgba(46, 134, 171, 0.2)',
        hovertemplate='<b>Distance:</b> %{x:.1f}px<br><b>Weight:</b> %{y:.3f}<extra></extra>',
        showlegend=False
    ))

    # Add median line
    if median_dist is not None:
        fig.add_vline(
            x=median_dist,
            line_dash="dash",
            line_color="#A23B72",
            line_width=2,
            annotation_text=f"Median: {median_dist:.1f}px",
            annotation_position="top",
            annotation_font_size=10,
            annotation_font_color="#A23B72"
        )

    # Add fewer key points for cleaner display
    key_points = [0.5, 0.9]
    colors = ['#E74C3C', '#27AE60']
    
    for i, point in enumerate(key_points):
        idx = np.argmin(np.abs(cumulative_values - point))
        if idx < len(distance_bins):
            fig.add_trace(go.Scatter(
                x=[distance_bins[idx]],
                y=[cumulative_values[idx]],
                mode='markers',
                marker=dict(size=8, color=colors[i], line=dict(width=1, color='white')),
                name=f'{int(point * 100)}%',
                showlegend=False,
                hovertemplate=f'<b>{int(point * 100)}% Attribution</b><br>Distance: %{{x:.1f}}px<extra></extra>'
            ))

    # Rectangular layout with properly centered title
    fig.update_layout(
        title=dict(
            text="<b>Cumulative Attribution Analysis</b>",
            x=0.5,  # Center horizontally
            y=0.92,  # Positioned to avoid toolbar collision
            xanchor='center',  # Ensure proper centering
            font=dict(size=15, color='#2C3E50')
        ),
        xaxis_title="<b>Distance (pixels)</b>",
        yaxis_title="<b>Cumulative Weight</b>",
        height=320,  # Slightly increased height for better title spacing
        font=dict(size=12),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        hovermode='x',
        margin=dict(t=60, b=60, l=80, r=40),  # Optimized margins
        showlegend=False
    )

    # Clean grid
    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    fig.update_yaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black',
        range=[0, 1.05]
    )

    return fig


def create_professional_attribution_histogram(attr_map):
    """Create rectangular attribution values distribution histogram with full title visibility"""
    fig = go.Figure()
    
    # Flatten attribution map
    attr_values = attr_map.flatten()
    
    # Remove very small values for cleaner visualization
    threshold = np.percentile(np.abs(attr_values), 5)
    significant_values = attr_values[np.abs(attr_values) > threshold]
    
    # Create histogram with custom bins
    fig.add_trace(go.Histogram(
        x=significant_values,
        nbinsx=40,
        name="Attribution Distribution",
        marker=dict(
            color='rgba(162, 59, 114, 0.7)',  # Different color for distinction
            line=dict(width=1, color='white')
        ),
        hovertemplate='<b>Attribution Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>',
        showlegend=False
    ))
    
    # Add vertical lines for statistics
    mean_val = np.mean(significant_values)
    median_val = np.median(significant_values)
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="#E74C3C",
        line_width=2,
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_position="top",
        annotation_font_size=9
    )
    
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="#27AE60",
        line_width=2,
        annotation_text=f"Median: {median_val:.4f}",
        annotation_position="bottom",
        annotation_font_size=9
    )
    
    # Rectangular layout with properly centered title
    fig.update_layout(
        title=dict(
            text="<b>Attribution Values Distribution</b>",
            x=0.5,  # Center horizontally
            y=0.92,  # Positioned to avoid toolbar collision
            xanchor='center',  # Ensure proper centering
            font=dict(size=15, color='#2C3E50')
        ),
        xaxis_title="<b>Attribution Value</b>",
        yaxis_title="<b>Frequency</b>",
        height=320,  # Slightly increased height for better title spacing
        font=dict(size=11),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=80, r=40),  # Optimized margins
        showlegend=False
    )
    
    # Grid
    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    fig.update_yaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=1,
        linecolor='black'
    )
    
    return fig


# --- Main Application ---

def main():
    st.title("🫁 Enhanced XAI Analysis for Lung Segmentation")
    st.markdown("---")

    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        page = st.radio(
            "Analysis Mode:",
            ["🔬 Enhanced Comparative Analysis", "📈 Training Analysis"],
            help="Select analysis mode"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Configuration")
        
        # Global settings
        show_debug = st.checkbox("🐛 Debug Mode", value=False, help="Show debug information")
        use_gpu_acceleration = st.checkbox("⚡ GPU Acceleration", value=True, help="Use GPU for computations")
        
        st.markdown("---")
        st.markdown("### 💡 Enhanced Features")
        if page == "🔬 Enhanced Comparative Analysis":
            st.markdown("""
            **🆕 New Features:**
            - Combined overlay visualization
            - Segmentation threshold slider
            - Enhanced attribution maps with grid centers
            - Vectorized performance optimization
            - Export functionality
            - Environment variable configuration
            - Advanced error handling
            """)
        else:
            st.markdown("""
            **📈 Training Analysis:**
            - Multi-model comparison
            - Interactive visualizations
            - Performance metrics
            - Best epoch identification
            """)

    # Main content based on page selection
    if page == "🔬 Enhanced Comparative Analysis":
        st.subheader("🔬 Enhanced Comparative XAI Analysis")

        # Check for available runs
        available_runs = safe_file_operation(get_available_runs)
        if not available_runs:
            st.error("❌ No evaluation results found. Please check your configuration and run experiments first.")
            
            # Configuration help
            with st.expander("⚙️ Configuration Help"):
                st.markdown("""
                **Environment Variables:**
                - `OUTPUT_DIR`: Path to outputs directory (default: ./outputs)
                - `DATA_PATH`: Path to datasets directory (default: /home/mohaisen_mohammed/Datasets)
                - `MONTGOMERY_PATH`: Path to Montgomery dataset
                - `JSRT_PATH`: Path to JSRT dataset
                
                **Config File:** Ensure config.yaml exists with proper dataset paths.
                """)
            return

        # Compact usage instructions
        with st.expander("📖 Usage Guide"):
            st.markdown("""
            1. **Select Models** in each column • 2. **Adjust Threshold** slider • 3. **Select Points** with sliders • 4. **Analyze** attribution • 5. **Export** results • 6. **Compare** across columns
            """)

        # Three-column layout with enhanced spacing
        col1, col2, col3 = st.columns(3, gap="large")

        all_column_data = []

        # Display enhanced columns
        with col1:
            with st.container():
                result1 = display_enhanced_column(1, available_runs)
                if result1:
                    all_column_data.append(result1)

        with col2:
            with st.container():
                result2 = display_enhanced_column(2, available_runs)
                if result2:
                    all_column_data.append(result2)

        with col3:
            with st.container():
                result3 = display_enhanced_column(3, available_runs)
                if result3:
                    all_column_data.append(result3)

        # Enhanced multi-model comparison
        if len(all_column_data) > 1:
            st.markdown("---")
            st.subheader("🔗 Advanced Multi-Model Comparison")

            try:
                # Create comparison plot
                fig = go.Figure()
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

                for i, item in enumerate(all_column_data):
                    color = colors[i % len(colors)]
                    hist_data = item["hist_data"]

                    if hist_data["distance_bins"] is not None and hist_data["cumulative_values"] is not None:
                        fig.add_trace(go.Scatter(
                            x=hist_data["distance_bins"],
                            y=hist_data["cumulative_values"],
                            mode='lines+markers',
                            name=f'{item["run_name"]} ({item["state_display"]})',
                            line=dict(color=color, width=4, shape='spline', smoothing=1.3),
                            marker=dict(size=6, color=color, line=dict(width=2, color='white')),
                            hovertemplate=f'<b>{item["run_name"]}</b><br>Distance: %{{x:.1f}} px<br>Cumulative Weight: %{{y:.3f}}<extra></extra>'
                        ))

                # Rectangular layout for comparison plot with centered title
                fig.update_layout(
                    title=dict(
                        text="<b>Multi-Model Comparison - Cumulative Attribution Analysis</b>",
                        x=0.5,  # Center horizontally
                        y=0.92,  # Positioned to avoid toolbar collision
                        xanchor='center',  # Ensure proper centering
                        font=dict(size=16, color='#2C3E50')
                    ),
                    xaxis_title="<b>Distance from Analysis Center (pixels)</b>",
                    yaxis_title="<b>Normalized Cumulative Attribution Weight</b>",
                    hovermode='x unified',
                    height=420,  # Slightly increased height for better spacing
                    yaxis=dict(range=[0, 1.05]),
                    font=dict(size=12),
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    margin=dict(t=70, b=80, l=80, r=200),  # Optimized top margin
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.88,  # Adjusted to avoid title overlap
                        xanchor="left",
                        x=1.02,
                        font=dict(size=11)
                    )
                )

                fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showline=True, 
                                linewidth=2, linecolor='black', mirror=True)
                fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showline=True, 
                                linewidth=2, linecolor='black', mirror=True)

                st.plotly_chart(fig, use_container_width=True)

                # Enhanced comparison table
                st.subheader("📊 Detailed Comparison Summary")
                stats_data = []
                for item in all_column_data:
                    stats_data.append({
                        "Model": item["run_name"],
                        "State": item["state_display"],
                        "Original Point": f"({item['center_point'][0]}, {item['center_point'][1]})",
                        "Analysis Point": f"({item['analysis_point'][0]}, {item['analysis_point'][1]})",
                        "Dice Score": f"{item['dice_score']:.4f}",
                        "Image Size": f"{item['original_size'][0]}×{item['original_size'][1]}",
                        "Threshold": f"{item['threshold']:.2f}",
                        "Dataset": item["model_info"]["dataset"]
                    })

                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

                # Export comparison results
                if st.button("📤 Export Comparison Results", type="secondary"):
                    comparison_data = {
                        'models': all_column_data,
                        'comparison_summary': stats_data,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    # Create timestamped export path
                    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                    export_path = f"./exports/comparison_results_{timestamp}"
                    
                    success, exported_files = export_analysis_results({'comparison_data': comparison_data}, export_path)
                    if success:
                        st.success(f"✅ Comparison results exported to {export_path}")
                        with st.expander("📁 Exported Files"):
                            for file in exported_files:
                                st.write(f"• {file}")
                        st.info(f"📂 Full path: {os.path.abspath(export_path)}")
                    else:
                        st.error("❌ Comparison export failed")

            except Exception as e:
                st.error(f"❌ Error creating comparison: {e}")
                if show_debug:
                    st.exception(e)

        elif len(all_column_data) == 1:
            st.info("💡 **Tip:** Configure models in multiple columns for advanced comparative analysis.")

    elif page == "📈 Training Analysis":
        # Enhanced training analysis (existing functionality)
        display_training_analysis()

    # Debug information
    if show_debug and st.sidebar.button("🔍 Show Debug Info"):
        with st.expander("🐛 Debug Information", expanded=False):
            st.json({
                "OUTPUT_DIR": OUTPUT_DIR,
                "DATA_PATH": DATA_PATH,
                "DATASETS": DATASETS,
                "Available Runs": available_runs if 'available_runs' in locals() else "Not loaded",
                "Session State Keys": list(st.session_state.keys())
            })


@st.cache_data
def get_available_runs():
    """Enhanced run detection with better error handling"""
    if not os.path.isdir(OUTPUT_DIR):
        return []

    runs = []
    try:
        for d in os.listdir(OUTPUT_DIR):
            run_dir = os.path.join(OUTPUT_DIR, d)
            if os.path.isdir(run_dir) and d.startswith('unet_'):
                # Check for either model files or evaluation results
                has_model = os.path.exists(os.path.join(run_dir, 'best_model.pth')) or \
                           os.path.exists(os.path.join(run_dir, 'final_model.pth'))
                has_eval = os.path.exists(os.path.join(run_dir, 'evaluation'))
                
                if has_model or has_eval:
                    runs.append(d)
                    
    except Exception as e:
        st.error(f"Error scanning runs: {e}")

    return sorted(runs)


def display_training_analysis():
    """Enhanced training analysis display"""
    st.header("📈 Professional Training Analysis")

    available_runs = get_available_runs()
    if not available_runs:
        st.warning("No training runs available for analysis.")
        return

    # Enhanced run selection
    selected_runs = st.multiselect(
        "Select runs to compare:",
        available_runs,
        default=available_runs[:6] if len(available_runs) > 6 else available_runs,
        format_func=lambda x: get_model_info(x)['display_name'],
        help="Select multiple runs for comparison"
    )

    if not selected_runs:
        st.info("Please select at least one run to analyze.")
        return

    # Training comparison plot with enhanced features
    fig = go.Figure()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#2A9D8F', '#E9C46A', '#F4A261']

    for i, run_name in enumerate(selected_runs):
        try:
            training_df = load_training_log(run_name)
            if training_df is not None:
                model_info = get_model_info(run_name)
                label = model_info['display_name']
                color = colors[i % len(colors)]

                # Add training and validation curves
                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['train_loss'],
                    mode='lines+markers',
                    name=f"{label} (Train)",
                    line=dict(color=color, dash='solid', width=3),
                    marker=dict(size=6, color=color, line=dict(width=2, color='white')),
                    hovertemplate=f'<b>{label} Training</b><br>Epoch: %{{x}}<br>Loss: %{{y:.6f}}<extra></extra>'
                ))

                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['val_loss'],
                    mode='lines+markers',
                    name=f"{label} (Val)",
                    line=dict(color=color, dash='dash', width=3),
                    marker=dict(size=6, color=color, line=dict(width=2, color='white')),
                    hovertemplate=f'<b>{label} Validation</b><br>Epoch: %{{x}}<br>Loss: %{{y:.6f}}<extra></extra>'
                ))

                # Mark best epoch
                best_epoch = training_df.loc[training_df['val_loss'].idxmin()]
                fig.add_vline(
                    x=best_epoch['epoch'],
                    line_dash="dot",
                    line_color=color,
                    line_width=2,
                    opacity=0.7,
                    annotation_text=f"{label} Best",
                    annotation_position="top",
                    annotation_font_size=10
                )

        except Exception as e:
            st.error(f"Error processing {run_name}: {e}")

    # Rectangular layout for training analysis with centered title
    fig.update_layout(
        title=dict(
            text="<b>Training Progress Comparison</b>",
            x=0.5,  # Center horizontally
            y=0.92,  # Positioned to avoid toolbar collision
            xanchor='center',  # Ensure proper centering
            font=dict(size=16, color='#2C3E50')
        ),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Loss</b>",
        hovermode='x unified',
        height=420,  # Slightly increased height for better spacing
        margin=dict(t=70, b=80, l=80, r=150),  # Optimized top margin
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(size=12),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.88,  # Adjusted to avoid title overlap
            xanchor="left",
            x=1.02,
            font=dict(size=10)
        )
    )

    fig.update_xaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', 
                     showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', 
                     showline=True, linewidth=2, linecolor='black', mirror=True)

    st.plotly_chart(fig, use_container_width=True)


@st.cache_data  
def load_training_log(run_name):
    """Load training log with caching"""
    log_path = os.path.join(OUTPUT_DIR, run_name, "training_log.csv")
    if os.path.exists(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception as e:
            st.error(f"Error loading training log: {e}")
    return None


if __name__ == "__main__":
    # Enhanced styling
    st.markdown("""
    <style>
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Enhanced sidebar */
        .css-1d391kg {
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            border-right: 2px solid #dee2e6;
        }

        /* Content area */
        .css-1y4p8pa {
            max-width: calc(100% - 220px) !important;
            padding-left: 1rem;
        }

        /* Enhanced typography */
        h1 {
            color: #2C3E50;
            font-weight: 700;
            font-size: 2.2rem !important;
            text-align: center;
            margin-bottom: 0.5rem !important;
            margin-top: 0.5rem !important;
        }
        
        h2, h3 {
            color: #34495E;
            font-weight: 600;
        }

        /* Enhanced buttons */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.75rem 1.5rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 16px rgba(46, 134, 171, 0.3) !important;
        }

        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 500 !important;
        }

        /* Enhanced sliders */
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%) !important;
            height: 8px !important;
            border-radius: 4px !important;
        }

        .stSlider > div > div > div > div > div {
            background-color: white !important;
            border: 3px solid #2E86AB !important;
            border-radius: 50% !important;
            box-shadow: 0 3px 8px rgba(0, 0, 0, 0.3) !important;
            width: 20px !important;
            height: 20px !important;
        }

        /* Enhanced metrics */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
            transition: transform 0.2s ease;
        }

        [data-testid="metric-container"]:hover {
            transform: translateY(-2px);
        }

        [data-testid="metric-container"] > div {
            color: white;
            font-weight: 600;
        }

        /* Enhanced info boxes */
        .stInfo {
            background: linear-gradient(135deg, #E8F6F3 0%, #D5F2EC 100%);
            border: 2px solid #2E86AB;
            border-radius: 12px;
            padding: 1.2rem;
            box-shadow: 0 4px 12px rgba(46, 134, 171, 0.1);
        }

        /* Enhanced expanders */
        .streamlit-expanderHeader {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            font-weight: 600;
            border: 1px solid #dee2e6;
        }

        /* Enhanced containers */
        .stContainer {
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            border: 1px solid #f0f0f0;
        }

        /* Enhanced DataFrames */
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
            border: 1px solid #E1E5E9;
        }

        /* Progress bars */
        .stProgress .st-bo {
            background: linear-gradient(90deg, #2E86AB 0%, #A23B72 100%);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.markdown("🔧 Please check your configuration and try again.")
        
        # Show error details in debug mode
        if st.sidebar.checkbox("Show Error Details", value=False):
            st.exception(e)