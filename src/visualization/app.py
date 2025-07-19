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

# --- Page Configuration ---
st.set_page_config(
    page_title=" XAI Lung Segmentation Analysis",
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
    OUTPUT_DIR = "./outputs"
    DATASETS = {}


# --- Enhanced XAI Analysis Functions ---
def calculate_weighted_median_distance(attr_map, center_x, center_y, min_abs_attr=None, max_abs_attr=None):
    """Calculate weighted median distance based on attribution values"""
    attr_z = attr_map.copy()

    if min_abs_attr is None:
        min_abs_attr = np.percentile(np.abs(attr_z), 5)  # More conservative threshold
    if max_abs_attr is None:
        max_abs_attr = np.percentile(np.abs(attr_z), 95)

    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)

    if np.any(selected_mask):
        selected_indices = np.where(selected_mask)
        distances = np.sqrt((selected_indices[0] - center_y) ** 2 + (selected_indices[1] - center_x) ** 2)
        weights = np.abs(attr_z[selected_mask])

        if len(distances) > 0:
            sorted_idx = np.argsort(distances)
            sorted_distances = distances[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1] if len(cumulative_weights) > 0 else 0

            if total_weight > 0:
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0]
                if len(median_idx) > 0:
                    median_distance = sorted_distances[median_idx[0]]
                    return median_distance, sorted_distances, cumulative_weights, total_weight

    return None, None, None, None


def calculate_enhanced_cumulative_histogram(attr_map, center_x, center_y):
    """Enhanced cumulative histogram calculation with smoothing"""
    y_coords, x_coords = np.mgrid[0:attr_map.shape[0], 0:attr_map.shape[1]]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    weights = np.abs(attr_map)

    distances_flat = distances.flatten()
    weights_flat = weights.flatten()

    # Remove very small weights to reduce noise
    threshold = np.percentile(weights_flat, 5)
    significant_mask = weights_flat > threshold
    distances_clean = distances_flat[significant_mask]
    weights_clean = weights_flat[significant_mask]

    if len(distances_clean) == 0:
        return None, None

    sorted_idx = np.argsort(distances_clean)
    sorted_distances = distances_clean[sorted_idx]
    sorted_weights = weights_clean[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)

    if cumulative_weights[-1] > 0:
        normalized_cumulative = cumulative_weights / cumulative_weights[-1]
    else:
        normalized_cumulative = cumulative_weights

    # Create smoother binning
    max_dist = int(np.ceil(sorted_distances.max()))
    distance_bins = np.linspace(0, max_dist, min(max_dist + 1, 200))  # Smoother binning

    binned_cumulative = np.interp(distance_bins, sorted_distances, normalized_cumulative)

    return distance_bins, binned_cumulative


# --- Helper Functions ---
def parse_run_name(run_name):
    """Parse experiment run name into components"""
    parts = run_name.replace("unet_", "").split('_')
    if len(parts) >= 3:
        return {
            "dataset": parts[0].title(),
            "data_size": parts[1].title(),
            "epochs": str(parts[2])
        }
    return {"dataset": "unknown", "data_size": "unknown", "epochs": "unknown"}


def safe_file_check(filepath):
    """Safely check if file exists"""
    try:
        return os.path.exists(filepath)
    except:
        return False


# --- Data Loading Functions (with Caching) ---
@st.cache_data
def get_available_runs():
    """Get list of available experiment runs"""
    if not os.path.isdir(OUTPUT_DIR):
        return []

    runs = []
    try:
        for d in os.listdir(OUTPUT_DIR):
            run_dir = os.path.join(OUTPUT_DIR, d)
            if os.path.isdir(run_dir) and d.startswith('unet_'):
                has_final_model = safe_file_check(os.path.join(run_dir, 'final_model.pth'))
                has_training_log = safe_file_check(os.path.join(run_dir, 'training_log.csv'))

                if has_final_model or has_training_log:
                    runs.append(d)
    except Exception as e:
        st.error(f"Error scanning runs: {e}")

    return sorted(runs)


@st.cache_data
def get_available_states(run_name):
    """Get available evaluation states for a run"""
    eval_dir = os.path.join(OUTPUT_DIR, run_name, "evaluation")
    if not safe_file_check(eval_dir):
        return []

    states = []
    try:
        for state in ['underfitting', 'good_fitting', 'overfitting']:
            state_dir = os.path.join(eval_dir, state)
            if safe_file_check(state_dir):
                states.append(state)
    except:
        pass

    return states


@st.cache_data
def load_run_data(run_name, state, split):
    """Load evaluation results for a specific run, state, and split"""
    summary_path = os.path.join(OUTPUT_DIR, run_name, "evaluation", state, split, "_evaluation_summary.json")
    if not safe_file_check(summary_path):
        return None

    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {summary_path}: {e}")
        return None


@st.cache_data
def load_npz_data(npz_path):
    """Load XAI results from NPZ file"""
    if not safe_file_check(npz_path):
        return None

    try:
        with np.load(npz_path) as data:
            return {key: data[key] for key in data}
    except Exception as e:
        st.error(f"Error loading NPZ file: {e}")
        return None


@st.cache_data
def load_training_log(run_name):
    """Load training log for a run"""
    log_path = os.path.join(OUTPUT_DIR, run_name, "training_log.csv")
    if safe_file_check(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception as e:
            st.error(f"Error loading training log: {e}")
    return None


def get_original_image_path(image_name, dataset_name):
    """Get path to original UNPROCESSED image file"""
    if dataset_name.lower() in DATASETS:
        dataset_config = DATASETS[dataset_name.lower()]
        base_path = os.path.join(dataset_config['path'], dataset_config['images'])
        return os.path.join(base_path, image_name)

    if dataset_name.lower() == 'montgomery':
        return os.path.join("/home/mohaisen_mohammed/Datasets/MontgomeryDataset/CXR_png/", image_name)
    else:
        return os.path.join("/home/mohaisen_mohammed/Datasets/JSRT/images/", image_name)


def load_original_image(image_path):
    """Load original unprocessed X-ray image with proper scaling"""
    try:
        # Load original image without any preprocessing
        original_image = Image.open(image_path).convert("L")
        original_array = np.array(original_image)

        # Get original dimensions
        original_height, original_width = original_array.shape

        return original_array, (original_width, original_height)
    except Exception as e:
        st.error(f"Error loading original image: {e}")
        return None, None


def map_coordinates_to_analysis_space(x, y, original_size, analysis_size=(256, 256)):
    """Map coordinates from original image space to analysis space (256x256)"""
    orig_w, orig_h = original_size
    anal_w, anal_h = analysis_size

    # Scale coordinates
    mapped_x = int((x / orig_w) * anal_w)
    mapped_y = int((y / orig_h) * anal_h)

    # Ensure coordinates are within bounds
    mapped_x = max(0, min(mapped_x, anal_w - 1))
    mapped_y = max(0, min(mapped_y, anal_h - 1))

    return mapped_x, mapped_y


def create_clickable_image_selector(image_array, current_x, current_y, original_size, column_id):
    """Create an interactive image where users can click to select coordinates"""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Display the original image
    ax.imshow(image_array, cmap='gray', extent=[0, original_size[0], original_size[1], 0])

    # Add professional crosshair
    ax.axhline(y=current_y, color='#FF4444', linestyle='-', alpha=0.8, linewidth=3)
    ax.axvline(x=current_x, color='#FF4444', linestyle='-', alpha=0.8, linewidth=3)

    # Add target marker with multiple visual elements
    circle1 = Circle((current_x, current_y), radius=15, color='#FF4444', fill=False, linewidth=4)
    circle2 = Circle((current_x, current_y), radius=10, color='white', fill=False, linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.plot(current_x, current_y, 'o', color='#FF4444', markersize=8,
            markerfacecolor='white', markeredgecolor='#FF4444', markeredgewidth=3)

    # Add subtle grid for reference
    grid_spacing = max(original_size[0] // 8, original_size[1] // 8)
    for i in range(0, original_size[1], grid_spacing):
        ax.axhline(y=i, color='white', alpha=0.1, linewidth=0.5)
    for j in range(0, original_size[0], grid_spacing):
        ax.axvline(x=j, color='white', alpha=0.1, linewidth=0.5)

    ax.set_title(f"🎯 Click to Select Analysis Point\nCurrent: ({current_x}, {current_y})",
                 fontsize=16, pad=20, fontweight='bold', color='#2C3E50')
    ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

    ax.set_xlim(0, original_size[0])
    ax.set_ylim(original_size[1], 0)  # Invert Y axis for image coordinates

    plt.tight_layout()
    return fig


def display_professional_attribution_map(attr_map, center_x, center_y, title, numerical_values=None):
    """Display professional attribution map with clean visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Apply Gaussian smoothing to reduce noise
    from scipy import ndimage
    smoothed_attr = ndimage.gaussian_filter(attr_map, sigma=1.0)

    # Enhanced color normalization
    vmax = np.percentile(np.abs(smoothed_attr), 98)
    vmin = -vmax

    # Main attribution map with professional colormap
    im1 = ax1.imshow(smoothed_attr, cmap='RdBu_r', interpolation='bilinear',
                     vmin=vmin, vmax=vmax, alpha=0.9)

    # Professional crosshair
    ax1.axhline(y=center_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
    ax1.axvline(x=center_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

    # Target marker
    circle = Circle((center_x, center_y), radius=8, color='#00FF00', fill=False, linewidth=3)
    ax1.add_patch(circle)
    ax1.plot(center_x, center_y, 'o', color='#00FF00', markersize=6,
             markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

    ax1.set_title(f"{title} - Smoothed View\n🎯 Analysis Point: ({center_x}, {center_y})",
                  fontsize=14, fontweight='bold', color='#2C3E50')
    ax1.set_xlabel("X Coordinate", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Y Coordinate", fontsize=12, fontweight='bold')

    # Colorbar for main map
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Attribution Score\n🔴 Positive | 🔵 Negative',
                    rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Raw attribution map for comparison
    im2 = ax2.imshow(attr_map, cmap='RdBu_r', interpolation='nearest',
                     vmin=vmin, vmax=vmax, alpha=0.9)

    ax2.axhline(y=center_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
    ax2.axvline(x=center_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

    circle2 = Circle((center_x, center_y), radius=8, color='#00FF00', fill=False, linewidth=3)
    ax2.add_patch(circle2)
    ax2.plot(center_x, center_y, 'o', color='#00FF00', markersize=6,
             markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

    ax2.set_title(f"{title} - Raw Data\n🎯 Analysis Point: ({center_x}, {center_y})",
                  fontsize=14, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel("X Coordinate", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Y Coordinate", fontsize=12, fontweight='bold')

    # Colorbar for raw map
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('Attribution Score\n🔴 Positive | 🔵 Negative',
                    rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Add statistical information
    pos_attr = np.sum(attr_map[attr_map > 0])
    neg_attr = np.sum(attr_map[attr_map < 0])
    max_pos = np.max(attr_map)
    min_neg = np.min(attr_map)
    center_value = attr_map[center_y, center_x] if 0 <= center_y < attr_map.shape[0] and 0 <= center_x < attr_map.shape[
        1] else 0

    stats_text = (f"Center Value: {center_value:.6f}\n"
                  f"Max Positive: {max_pos:.6f}\n"
                  f"Min Negative: {min_neg:.6f}\n"
                  f"Total Positive: {pos_attr:.3f}\n"
                  f"Total Negative: {neg_attr:.3f}")

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    return fig


def create_professional_cumulative_plot(distance_bins, cumulative_values, center_point, median_dist=None):
    """Create a professional, smooth cumulative attribution plot"""
    fig = go.Figure()

    # Create smooth line with gradient fill
    fig.add_trace(go.Scatter(
        x=distance_bins,
        y=cumulative_values,
        mode='lines',
        name='Cumulative Attribution',
        line=dict(color='#2E86AB', width=4, shape='spline', smoothing=1.3),
        fill='tonexty',
        fillcolor='rgba(46, 134, 171, 0.2)',
        hovertemplate='<b>Distance:</b> %{x:.1f} pixels<br><b>Cumulative Weight:</b> %{y:.3f}<extra></extra>'
    ))

    # Add median line if available
    if median_dist is not None:
        fig.add_vline(
            x=median_dist,
            line_dash="dash",
            line_color="#A23B72",
            line_width=3,
            annotation_text=f"Weighted Median: {median_dist:.1f}px",
            annotation_position="top right",
            annotation_font_size=12,
            annotation_font_color="#A23B72"
        )

    # Add markers at key points
    key_points = [0.25, 0.5, 0.75, 0.9]
    for point in key_points:
        idx = np.argmin(np.abs(cumulative_values - point))
        if idx < len(distance_bins):
            fig.add_trace(go.Scatter(
                x=[distance_bins[idx]],
                y=[cumulative_values[idx]],
                mode='markers',
                marker=dict(size=10, color='#F18F01', line=dict(width=2, color='white')),
                name=f'{int(point * 100)}% Mark',
                showlegend=False,
                hovertemplate=f'<b>{int(point * 100)}% Attribution</b><br>Distance: %{{x:.1f}} pixels<extra></extra>'
            ))

    fig.update_layout(
        title=dict(
            text=f"<b>📈 Cumulative Attribution Analysis</b><br><sub>From Point ({center_point[0]}, {center_point[1]})</sub>",
            x=0.5,
            font=dict(size=18, color='#2C3E50')
        ),
        xaxis_title="<b>Distance from Analysis Center (pixels)</b>",
        yaxis_title="<b>Normalized Cumulative Attribution Weight</b>",
        height=500,
        font=dict(size=14),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        hovermode='x',
        showlegend=False
    )

    # Enhanced grid and styling
    fig.update_xaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True
    )
    fig.update_yaxes(
        gridcolor='rgba(0,0,0,0.1)',
        gridwidth=1,
        showline=True,
        linewidth=2,
        linecolor='black',
        mirror=True,
        range=[0, 1.05]
    )

    return fig


def display_column(column_id, available_runs):
    """Display analysis column with click-based coordinate selection"""
    st.markdown(f"### 🔬 Analysis Column {column_id}")

    # Run selection
    selected_run = st.selectbox(
        f"Select Model Run",
        [""] + available_runs,
        key=f"run_{column_id}",
        format_func=lambda name: " | ".join(parse_run_name(name).values()) if name else "Select a run..."
    )

    if not selected_run:
        st.info("Please select a model run.")
        return None

    # State selection
    available_states = get_available_states(selected_run)
    if not available_states:
        st.warning(f"No evaluation results found for {selected_run}")
        return None

    selected_state = st.selectbox(
        f"Select Model State",
        available_states,
        key=f"state_{column_id}",
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Split selection
    selected_split = st.selectbox(
        f"Select Data Split",
        ["test", "validation", "training"],
        key=f"split_{column_id}"
    )

    # Load data
    with st.spinner("Loading data..."):
        run_data = load_run_data(selected_run, selected_state, selected_split)

    if not run_data:
        st.error(f"Could not load data for {selected_run}/{selected_state}/{selected_split}")
        return None

    # Image selection
    results = run_data.get("per_sample_results", [])
    if not results:
        st.warning("No sample results found.")
        return None

    sorted_results = sorted(results, key=lambda x: x.get("dice_score", 0), reverse=True)

    selected_image_data = st.selectbox(
        f"Select Image",
        sorted_results,
        key=f"image_{column_id}",
        format_func=lambda x: f"{x['image_name']} | Dice: {x['dice_score']:.4f} | IoU: {x['iou_score']:.4f}"
    )

    if not selected_image_data:
        st.warning("No images available.")
        return None

    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Dice Score", f"{selected_image_data['dice_score']:.4f}")
    with col2:
        st.metric("📐 IoU Score", f"{selected_image_data['iou_score']:.4f}")

    # Load XAI data
    with st.spinner("Loading XAI data..."):
        npz_data = load_npz_data(selected_image_data["xai_results_path"])

    if not npz_data:
        st.error(f"Failed to load XAI data for {selected_image_data['image_name']}")
        return None

    # Display basic information
    parsed_name = parse_run_name(selected_run)
    epoch_used = run_data.get('epoch_used', 'N/A')

    st.markdown(f"**{parsed_name['dataset']} | {parsed_name['data_size']} | {parsed_name['epochs']} epochs**")
    st.markdown(
        f"State: `{selected_state.replace('_', ' ').title()}` | Split: `{selected_split.upper()}` | Epoch: `{epoch_used}`")

    # Load ORIGINAL unprocessed image
    try:
        original_image_path = get_original_image_path(
            selected_image_data['image_name'],
            parsed_name['dataset']
        )

        original_array, original_size = load_original_image(original_image_path)
        if original_array is None:
            st.error("Could not load original image")
            return None

        gt_mask = npz_data['ground_truth']

    except Exception as e:
        st.error(f"Error loading images: {e}")
        return None

    # ROW 1: Original X-ray and Ground Truth
    st.markdown("🖼️ Original  Image and Ground Truth:")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original_array, caption=f"Original X-ray ({original_size[0]}x{original_size[1]})",
                 use_container_width=True, clamp=True)

    with col2:
        st.image(gt_mask, caption="Ground Truth (256x256)", use_container_width=True, clamp=True)

    # ROW 2: Interactive Coordinate Selection
    st.markdown("**🎯 Interactive Coordinate Selection:**")

    # Initialize coordinates in session state
    if f"orig_x_{column_id}" not in st.session_state:
        st.session_state[f"orig_x_{column_id}"] = original_size[0] // 2
    if f"orig_y_{column_id}" not in st.session_state:
        st.session_state[f"orig_y_{column_id}"] = original_size[1] // 2

    # Coordinate input with sliders for full image coverage
    coord_col1, coord_col2, coord_col3 = st.columns([1, 1, 1])

    with coord_col1:
        orig_x = st.slider(
            "🔄 X Coordinate (Original)",
            0, original_size[0] - 1,
            st.session_state[f"orig_x_{column_id}"],
            key=f"orig_x_slider_{column_id}",
            help=f"Select X coordinate (0 to {original_size[0] - 1})"
        )

    with coord_col2:
        orig_y = st.slider(
            "🔄 Y Coordinate (Original)",
            0, original_size[1] - 1,
            st.session_state[f"orig_y_{column_id}"],
            key=f"orig_y_slider_{column_id}",
            help=f"Select Y coordinate (0 to {original_size[1] - 1})"
        )

    with coord_col3:
        if st.button("🎯 Reset to Center", key=f"center_{column_id}"):
            st.session_state[f"orig_x_{column_id}"] = original_size[0] // 2
            st.session_state[f"orig_y_{column_id}"] = original_size[1] // 2
            st.rerun()

    # Update session state
    st.session_state[f"orig_x_{column_id}"] = orig_x
    st.session_state[f"orig_y_{column_id}"] = orig_y

    # Map coordinates to analysis space
    mapped_x, mapped_y = map_coordinates_to_analysis_space(orig_x, orig_y, original_size)

    # Display coordinate information
    st.info(
        f"🎯 **Original Coordinates:** X={orig_x}, Y={orig_y} | **Analysis Coordinates:** X={mapped_x}, Y={mapped_y}")

    # Professional Analyze Button
    analyze_button = st.button(
        "🔬 Analyze Selected Point",
        key=f"analyze_{column_id}",
        help="Click to run Integrated Gradients analysis on the selected point",
        type="primary"
    )

    # Display clickable image selector
    try:
        fig = create_clickable_image_selector(original_array, orig_x, orig_y, original_size, column_id)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    except Exception as e:
        st.error(f"Error displaying image selector: {e}")

    # ROW 3: Analysis Results (shown when button is clicked or automatically)
    if analyze_button or f"analysis_done_{column_id}" in st.session_state:
        st.session_state[f"analysis_done_{column_id}"] = True

        st.markdown("🧠 Integrated Gradients Analysis:")
        try:
            if 'ig_map' in npz_data:
                # Display professional attribution map
                fig = display_professional_attribution_map(
                    npz_data['ig_map'],
                    mapped_x, mapped_y,
                    "Integrated Gradients Attribution"
                )
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Calculate metrics
                median_dist, sorted_distances, cumulative_weights, total_weight = calculate_weighted_median_distance(
                    npz_data['ig_map'], mapped_x, mapped_y
                )

                # Display numerical values
                st.markdown("**📊 Numerical Analysis Results:**")

                metric_col1, metric_col2, metric_col3 = st.columns(3)

                with metric_col1:
                    center_value = npz_data['ig_map'][mapped_y, mapped_x] if 0 <= mapped_y < npz_data['ig_map'].shape[
                        0] and 0 <= mapped_x < npz_data['ig_map'].shape[1] else 0
                    st.metric("🎯 Center Attribution", f"{center_value:.6f}")

                with metric_col2:
                    if median_dist is not None:
                        st.metric("📏 Weighted Median Distance", f"{median_dist:.2f} px")
                    else:
                        st.metric("📏 Weighted Median Distance", "N/A")

                with metric_col3:
                    max_attribution = np.max(np.abs(npz_data['ig_map']))
                    st.metric("📈 Max Attribution", f"{max_attribution:.6f}")

                # Enhanced cumulative histogram with professional styling
                distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(
                    npz_data['ig_map'], mapped_x, mapped_y
                )

                if distance_bins is not None:
                    # Create professional cumulative plot
                    fig = create_professional_cumulative_plot(
                        distance_bins, cumulative_values, (orig_x, orig_y), median_dist
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Display detailed numerical values
                    st.markdown("**🔢 Detailed Numerical Values:**")

                    # Create expandable section for detailed data
                    with st.expander("📋 View Detailed Attribution Data", expanded=False):
                        # Attribution statistics around the selected point
                        radius = 10
                        y_start = max(0, mapped_y - radius)
                        y_end = min(npz_data['ig_map'].shape[0], mapped_y + radius + 1)
                        x_start = max(0, mapped_x - radius)
                        x_end = min(npz_data['ig_map'].shape[1], mapped_x + radius + 1)

                        local_patch = npz_data['ig_map'][y_start:y_end, x_start:x_end]

                        detail_col1, detail_col2 = st.columns(2)

                        with detail_col1:
                            st.markdown("**Local Region Statistics (±10px):**")
                            st.write(f"• Mean Attribution: {np.mean(local_patch):.6f}")
                            st.write(f"• Std Attribution: {np.std(local_patch):.6f}")
                            st.write(f"• Min Attribution: {np.min(local_patch):.6f}")
                            st.write(f"• Max Attribution: {np.max(local_patch):.6f}")

                        with detail_col2:
                            st.markdown("**Global Image Statistics:**")
                            st.write(f"• Global Mean: {np.mean(npz_data['ig_map']):.6f}")
                            st.write(f"• Global Std: {np.std(npz_data['ig_map']):.6f}")
                            st.write(f"• Global Min: {np.min(npz_data['ig_map']):.6f}")
                            st.write(f"• Global Max: {np.max(npz_data['ig_map']):.6f}")

                        # Histogram of attribution values
                        st.markdown("**Attribution Value Distribution:**")
                        hist_fig = go.Figure()
                        hist_fig.add_trace(go.Histogram(
                            x=npz_data['ig_map'].flatten(),
                            nbinsx=50,
                            name="Attribution Distribution",
                            marker_color='rgba(46, 134, 171, 0.7)',
                            marker_line=dict(width=1, color='white')
                        ))
                        hist_fig.update_layout(
                            title="Distribution of Attribution Values",
                            xaxis_title="Attribution Value",
                            yaxis_title="Frequency",
                            height=300
                        )
                        st.plotly_chart(hist_fig, use_container_width=True)

            else:
                st.warning("Integrated Gradients data not available")
        except Exception as e:
            st.error(f"Error displaying Integrated Gradients: {e}")

    # ROW 4: Uncertainty Analysis
    st.markdown("**🔬 Uncertainty Analysis:**")
    try:
        if 'uncertainty_map' in npz_data:
            fig, ax = plt.subplots(figsize=(12, 10))

            im = ax.imshow(npz_data['uncertainty_map'], cmap='plasma', interpolation='bilinear', alpha=0.9)

            # Add crosshair at mapped coordinates
            ax.axhline(y=mapped_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
            ax.axvline(x=mapped_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

            circle = Circle((mapped_x, mapped_y), radius=8, color='#00FF00', fill=False, linewidth=3)
            ax.add_patch(circle)
            ax.plot(mapped_x, mapped_y, 'o', color='#00FF00', markersize=6,
                    markerfacecolor='white', markeredgecolor='#00FF00', markeredgewidth=2)

            ax.set_title(f"Model Prediction Uncertainty\n🎯 Analysis Point: ({mapped_x}, {mapped_y})",
                         fontsize=16, pad=20, fontweight='bold', color='#2C3E50')
            ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
            ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
            cbar.set_label('Uncertainty Level\n🟡 High | 🟣 Low',
                           rotation=270, labelpad=25, fontsize=14, fontweight='bold')

            # Add uncertainty statistics
            uncertainty_at_point = npz_data['uncertainty_map'][mapped_y, mapped_x] if 0 <= mapped_y < \
                                                                                      npz_data['uncertainty_map'].shape[
                                                                                          0] and 0 <= mapped_x < \
                                                                                      npz_data['uncertainty_map'].shape[
                                                                                          1] else 0
            mean_uncertainty = np.mean(npz_data['uncertainty_map'])
            max_uncertainty = np.max(npz_data['uncertainty_map'])

            stats_text = (f"Point Uncertainty: {uncertainty_at_point:.4f}\n"
                          f"Mean Uncertainty: {mean_uncertainty:.4f}\n"
                          f"Max Uncertainty: {max_uncertainty:.4f}")

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax.set_xlim(0, 256)
            ax.set_ylim(256, 0)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.warning("Uncertainty map not available")
    except Exception as e:
        st.error(f"Error displaying uncertainty map: {e}")

    # ROW 5: Training History
    st.markdown("**📈 Training History:**")
    try:
        training_df = load_training_log(selected_run)
        if training_df is not None:
            fig = go.Figure()

            # Add training loss
            fig.add_trace(go.Scatter(
                x=training_df['epoch'],
                y=training_df['train_loss'],
                mode='lines+markers',
                name='Training Loss',
                line=dict(color='#2E86AB', width=3),
                marker=dict(size=6, color='#2E86AB', line=dict(width=2, color='white')),
                hovertemplate='<b>Training Loss</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
            ))

            # Add validation loss
            fig.add_trace(go.Scatter(
                x=training_df['epoch'],
                y=training_df['val_loss'],
                mode='lines+markers',
                name='Validation Loss',
                line=dict(color='#A23B72', width=3),
                marker=dict(size=6, color='#A23B72', line=dict(width=2, color='white')),
                hovertemplate='<b>Validation Loss</b><br>Epoch: %{x}<br>Loss: %{y:.6f}<extra></extra>'
            ))

            # Mark best epoch
            best_epoch = training_df.loc[training_df['val_loss'].idxmin()]
            fig.add_vline(
                x=best_epoch['epoch'],
                line_dash="dash",
                line_color="#F18F01",
                line_width=3,
                annotation_text=f"Best Model (Epoch {int(best_epoch['epoch'])})",
                annotation_position="top"
            )

            fig.update_layout(
                title=dict(
                    text=f"<b>Training Progress - {parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}ep</b>",
                    x=0.5,
                    font=dict(size=16, color='#2C3E50')
                ),
                xaxis_title="<b>Epoch</b>",
                yaxis_title="<b>Loss</b>",
                hovermode='x unified',
                height=400,
                margin=dict(t=80, b=60, l=60, r=60),
                plot_bgcolor='rgba(248, 249, 250, 0.8)',
                paper_bgcolor='white',
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            fig.update_xaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', showline=True, linewidth=2,
                             linecolor='black')
            fig.update_yaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', showline=True, linewidth=2,
                             linecolor='black')

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Training log not available")
    except Exception as e:
        st.error(f"Error displaying training curve: {e}")

    # Return data for combined analysis
    try:
        if f"analysis_done_{column_id}" in st.session_state:
            distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(
                npz_data['ig_map'], mapped_x, mapped_y
            )

            return {
                "run_name": f"Col {column_id}: {parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}",
                "hist_data": {"distance_bins": distance_bins, "cumulative_values": cumulative_values},
                "state": selected_state,
                "epoch_used": epoch_used,
                "center_point": (orig_x, orig_y),
                "analysis_point": (mapped_x, mapped_y),
                "dice_score": selected_image_data['dice_score'],
                "original_size": original_size
            }
    except Exception as e:
        st.error(f"Error calculating histogram: {e}")

    return None


def display_training_analysis():
    """Display enhanced training curve analysis"""
    st.header("📈 Professional Training Analysis")

    available_runs = get_available_runs()
    if not available_runs:
        st.warning("No runs available for analysis.")
        return

    # Multi-select for comparing multiple runs
    selected_runs = st.multiselect(
        "Select runs to compare:",
        available_runs,
        default=available_runs[:6] if len(available_runs) > 6 else available_runs,
        format_func=lambda x: " | ".join(parse_run_name(x).values()),
        help="Select multiple runs to compare their training progress"
    )

    if not selected_runs:
        st.info("Please select at least one run to analyze.")
        return

    # Create enhanced comparison plot
    fig = go.Figure()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7', '#2A9D8F', '#E9C46A', '#F4A261']

    for i, run_name in enumerate(selected_runs):
        try:
            training_df = load_training_log(run_name)
            if training_df is not None:
                parsed = parse_run_name(run_name)
                label = f"{parsed['dataset']} {parsed['data_size']} {parsed['epochs']}ep"
                color = colors[i % len(colors)]

                # Training curve
                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['train_loss'],
                    mode='lines+markers',
                    name=f"{label} (Train)",
                    line=dict(color=color, dash='solid', width=3),
                    marker=dict(size=6, color=color, line=dict(width=2, color='white')),
                    hovertemplate=f'<b>{label} Training</b><br>Epoch: %{{x}}<br>Loss: %{{y:.6f}}<extra></extra>'
                ))

                # Validation curve
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

    fig.update_layout(
        title=dict(
            text="<b>📊 Comprehensive Training Progress Comparison</b>",
            x=0.5,
            font=dict(size=20, color='#2C3E50')
        ),
        xaxis_title="<b>Epoch</b>",
        yaxis_title="<b>Loss</b>",
        hovermode='x unified',
        height=700,
        margin=dict(t=100, b=80, l=80, r=150),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(size=14),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=11)
        )
    )

    fig.update_xaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', showline=True, linewidth=2, linecolor='black',
                     mirror=True)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=1, griddash='dot', showline=True, linewidth=2, linecolor='black',
                     mirror=True)

    st.plotly_chart(fig, use_container_width=True)


# --- Main App ---
def main():
    st.title("🫁  XAI Analysis for Lung Segmentation")
    st.markdown("*Interactive Analysis with  Image Support and Visualizations*")
    st.markdown("---")

    # Very narrow sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Navigation")
        page = st.radio(
            "Analysis Mode:",
            ["🔬 Comparative Analysis", "📈 Training Analysis"],
            help="Select the type of analysis"
        )

        st.markdown("---")
        st.markdown("### 💡 Tips")
        if page == "🔬 Comparative Analysis":
            st.markdown("""
            **🎯 New Features:**
            - Original unprocessed images
            - Full coordinate coverage
            - Click-to-analyze functionality
            - Professional visualizations
            - Detailed numerical analysis
            """)
        else:
            st.markdown("""
            **📈 Training Analysis:**
            - Compare multiple models
            - Interactive hover details
            - Best epoch markers
            - Professional styling
            """)

    # Main content
    if page == "🔬 Comparative Analysis":
        st.header("🔬 Professional Comparative XAI Analysis")
        st.markdown("*Analyze original unprocessed X-ray images with expert-level attribution mapping*")

        available_runs = get_available_runs()
        if not available_runs:
            st.error("❌ No evaluation results found. Please run experiments first.")
            return

        st.info("""
         **How to Use:** 
        1. Select models in each column
        2. Use sliders to choose any pixel on the original image
        3. Click 'Analyze Selected Point' to run Integrated Gradients analysis
        4. View professional attribution maps and detailed numerical results
        """)

        # Three columns for comparison
        col1, col2, col3 = st.columns(3, gap="medium")

        all_column_data = []

        with col1:
            result1 = display_column(1, available_runs)
            if result1:
                all_column_data.append(result1)

        with col2:
            result2 = display_column(2, available_runs)
            if result2:
                all_column_data.append(result2)

        with col3:
            result3 = display_column(3, available_runs)
            if result3:
                all_column_data.append(result3)

        # Enhanced combined comparison
        if len(all_column_data) > 1:
            st.markdown("---")
            st.subheader("🔗 Professional Multi-Model Comparison")

            try:
                fig = go.Figure()
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']

                for i, item in enumerate(all_column_data):
                    color = colors[i % len(colors)]
                    hist_data = item["hist_data"]

                    if hist_data["distance_bins"] is not None and hist_data["cumulative_values"] is not None:
                        fig.add_trace(go.Scatter(
                            x=hist_data["distance_bins"],
                            y=hist_data["cumulative_values"],
                            mode='lines',
                            name=f'{item["run_name"]} ({item["state"]})',
                            line=dict(color=color, width=4, shape='spline', smoothing=1.3),
                            hovertemplate=f'<b>{item["run_name"]}</b><br>Distance: %{{x:.1f}} px<br>Cumulative Weight: %{{y:.3f}}<extra></extra>'
                        ))

                fig.update_layout(
                    title=dict(
                        text="<b>📊 Professional Comparison of Cumulative Weighted Attributions</b>",
                        x=0.5,
                        font=dict(size=18, color='#2C3E50')
                    ),
                    xaxis_title="<b>Distance from Analysis Center (pixels)</b>",
                    yaxis_title="<b>Normalized Cumulative Attribution Weight</b>",
                    hovermode='x',
                    height=600,
                    yaxis=dict(range=[0, 1.05]),
                    font=dict(size=14),
                    plot_bgcolor='rgba(248, 249, 250, 0.8)',
                    paper_bgcolor='white',
                    margin=dict(t=80, b=60, l=60, r=150)
                )

                fig.update_xaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showline=True, linewidth=2,
                                 linecolor='black', mirror=True)
                fig.update_yaxes(gridcolor='rgba(0,0,0,0.1)', gridwidth=1, showline=True, linewidth=2,
                                 linecolor='black', mirror=True)

                st.plotly_chart(fig, use_container_width=True)

                # Summary table
                st.subheader("📊 Detailed Model Comparison")
                stats_data = []
                for item in all_column_data:
                    stats_data.append({
                        "Model": item["run_name"],
                        "State": item["state"].replace('_', ' ').title(),
                        "Original Point": f"({item['center_point'][0]}, {item['center_point'][1]})",
                        "Analysis Point": f"({item['analysis_point'][0]}, {item['analysis_point'][1]})",
                        "Dice Score": f"{item['dice_score']:.4f}",
                        "Image Size": f"{item['original_size'][0]}x{item['original_size'][1]}"
                    })

                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"❌ Error creating comparison plot: {e}")

        elif len(all_column_data) == 1:
            st.info("💡 **Tip:** Select models in multiple columns for advanced comparative analysis.")

    elif page == "📈 Training Analysis":
        display_training_analysis()


if __name__ == "__main__":
    # Professional styling
    st.markdown("""
    <style>
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Narrow sidebar */
        .css-1d391kg {
            width: 180px !important;
            min-width: 180px !important;
            max-width: 180px !important;
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        /* Maximize content */
        .css-1y4p8pa {
            max-width: calc(100% - 200px) !important;
            padding-left: 1rem;
        }

        /* Typography */
        h1 {
            color: #2C3E50;
            font-weight: 700;
            font-size: 2.3rem !important;
        }
        h2, h3 {
            color: #34495E;
            font-weight: 600;
        }

        /* Professional button styling */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%) !important;
            border: none !important;
            border-radius: 8px !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
            font-size: 0.9rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        }

        .stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 12px rgba(46, 134, 171, 0.3) !important;
        }

        /* Slider enhancements */
        .stSlider > div > div > div > div {
            background-color: #2E86AB !important;
            height: 6px !important;
        }

        .stSlider > div > div > div > div > div {
            background-color: #A23B72 !important;
            border: 2px solid white !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
        }

        /* Metric styling */
        [data-testid="metric-container"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        [data-testid="metric-container"] > div {
            color: white;
        }

        /* Info boxes */
        .stInfo {
            background: linear-gradient(135deg, #E8F6F3 0%, #D5F2EC 100%);
            border: 1px solid #2E86AB;
            border-radius: 10px;
            padding: 1rem;
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: rgba(248, 249, 250, 0.8);
            border-radius: 8px;
            font-weight: 600;
        }

        /* DataFrame styling */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            border: 1px solid #E1E5E9;
        }
    </style>
    """, unsafe_allow_html=True)

    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.markdown("🔧 Please check your setup and try again.")