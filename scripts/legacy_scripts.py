import streamlit as st
import os
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import glob
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Import custom modules
from data_loader import LungDataset, ResizeAndToTensor
from model import UNet
from utils import get_data_splits, dice_score, iou_score, calculate_cumulative_histogram

# Set page config
st.set_page_config(
    page_title="Lung Segmentation XAI Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load configuration
@st.cache_data
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


# Load model
@st.cache_resource
def load_model(model_path, device):
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


# Load dataset
@st.cache_data
def load_dataset(dataset_name, config):
    return LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())


# Get available experiments
@st.cache_data
def get_available_experiments(config):
    experiments = []
    output_dir = config['output_base_dir']

    for exp_name in config['experiments'].keys():
        exp_dir = os.path.join(output_dir, exp_name)
        if os.path.exists(exp_dir):
            # Check if evaluation results exist
            eval_dir = os.path.join(exp_dir, 'evaluation')
            training_log = os.path.join(exp_dir, 'training_log.csv')

            exp_info = {
                'name': exp_name,
                'path': exp_dir,
                'has_training_log': os.path.exists(training_log),
                'has_evaluation': os.path.exists(eval_dir),
                'splits_available': []
            }

            # Check available splits
            if os.path.exists(eval_dir):
                for split in ['training', 'validation', 'test']:
                    split_dir = os.path.join(eval_dir, split)
                    if os.path.exists(split_dir):
                        exp_info['splits_available'].append(split)

            experiments.append(exp_info)

    return experiments


# Load training history
@st.cache_data
def load_training_history(exp_path):
    training_log_path = os.path.join(exp_path, 'training_log.csv')
    if os.path.exists(training_log_path):
        return pd.read_csv(training_log_path)
    return None


# Load evaluation results
@st.cache_data
def load_evaluation_results(exp_path, split):
    eval_path = os.path.join(exp_path, 'evaluation', split, '_evaluation_summary.json')
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            return json.load(f)
    return None


# Load XAI maps
@st.cache_data
def load_xai_maps(exp_path, split, image_name):
    maps_path = os.path.join(exp_path, 'evaluation', split, 'maps', f'{image_name}.npz')
    if os.path.exists(maps_path):
        return np.load(maps_path)
    return None


def plot_training_curves(training_df, title):
    """Plot training and validation loss curves"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f"Training Curves - {title}"]
    )

    # Add training loss
    fig.add_trace(
        go.Scatter(
            x=training_df['epoch'],
            y=training_df['train_loss'],
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        )
    )

    # Add validation loss
    fig.add_trace(
        go.Scatter(
            x=training_df['epoch'],
            y=training_df['val_loss'],
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red', width=2),
            marker=dict(size=4)
        )
    )

    # Mark best epoch
    best_epoch = training_df.loc[training_df['val_loss'].idxmin()]
    fig.add_vline(
        x=best_epoch['epoch'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Best Model (Epoch {int(best_epoch['epoch'])})"
    )

    fig.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=400
    )

    return fig


def categorize_models(experiments, config):
    """Categorize models into underfit, well-fit, and overfit based on training curves"""
    categorized = {'underfit': [], 'well_fit': [], 'overfit': []}

    for exp in experiments:
        if not exp['has_training_log']:
            continue

        training_df = load_training_history(exp['path'])
        if training_df is None or len(training_df) == 0:
            continue

        # Simple heuristic for categorization
        final_train_loss = training_df['train_loss'].iloc[-1]
        final_val_loss = training_df['val_loss'].iloc[-1]
        min_val_loss = training_df['val_loss'].min()

        # Calculate overfitting indicator
        overfit_ratio = final_val_loss / min_val_loss if min_val_loss > 0 else 1

        # Get experiment parameters
        exp_params = config['experiments'][exp['name']]
        epochs = exp_params['epochs']

        # Categorization logic
        if epochs <= 50 and final_train_loss > 0.1:
            categorized['underfit'].append(exp)
        elif overfit_ratio > 1.2:  # Validation loss increased significantly from minimum
            categorized['overfit'].append(exp)
        else:
            categorized['well_fit'].append(exp)

    return categorized


def display_model_comparison(models_dict, config):
    """Display side-by-side comparison of different model categories"""
    st.subheader("Model Comparison: Underfit vs Well-fit vs Overfit")

    cols = st.columns(3)

    categories = ['underfit', 'well_fit', 'overfit']
    category_titles = ['Underfit Models', 'Well-fit Models', 'Overfit Models']

    for i, (category, title) in enumerate(zip(categories, category_titles)):
        with cols[i]:
            st.markdown(f"### {title}")

            if not models_dict[category]:
                st.info(f"No {category} models available")
                continue

            # Select model from category
            model_names = [exp['name'] for exp in models_dict[category]]
            selected_model = st.selectbox(
                f"Select {category} model:",
                model_names,
                key=f"{category}_model_select"
            )

            if selected_model:
                # Find selected experiment
                selected_exp = next(exp for exp in models_dict[category] if exp['name'] == selected_model)

                # Display training curves
                training_df = load_training_history(selected_exp['path'])
                if training_df is not None:
                    fig = plot_training_curves(training_df, selected_model)
                    st.plotly_chart(fig, use_container_width=True)

                # Display evaluation metrics
                st.markdown("**Evaluation Metrics:**")
                for split in selected_exp['splits_available']:
                    eval_results = load_evaluation_results(selected_exp['path'], split)
                    if eval_results:
                        st.metric(
                            f"{split.title()} Dice Score",
                            f"{eval_results['average_dice_score']:.4f}"
                        )
                        st.metric(
                            f"{split.title()} IoU Score",
                            f"{eval_results['average_iou_score']:.4f}"
                        )


def display_sample_analysis(selected_experiments, config):
    """Display detailed analysis for selected samples"""
    st.subheader("Sample-wise Analysis")

    # Model selection
    model_names = [exp['name'] for exp in selected_experiments]
    selected_model = st.selectbox("Select Model for Analysis:", model_names)

    if not selected_model:
        return

    selected_exp = next(exp for exp in selected_experiments if exp['name'] == selected_model)

    # Split selection
    available_splits = selected_exp['splits_available']
    if not available_splits:
        st.warning("No evaluation results available for this model")
        return

    selected_split = st.selectbox("Select Data Split:", available_splits)

    # Load evaluation results
    eval_results = load_evaluation_results(selected_exp['path'], selected_split)
    if not eval_results:
        st.warning("No evaluation results found")
        return

    # Sample selection
    samples = eval_results['per_sample_results']
    sample_names = [sample['image_name'] for sample in samples]
    selected_sample = st.selectbox("Select Sample:", sample_names)

    if selected_sample:
        # Find sample data
        sample_data = next(sample for sample in samples if sample['image_name'] == selected_sample)

        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dice Score", f"{sample_data['dice_score']:.4f}")
        with col2:
            st.metric("IoU Score", f"{sample_data['iou_score']:.4f}")

        # Load and display XAI maps
        xai_data = load_xai_maps(selected_exp['path'], selected_split, selected_sample)
        if xai_data is not None:
            display_xai_visualizations(xai_data, selected_sample)


def display_xai_visualizations(xai_data, sample_name):
    """Display XAI visualizations including attribution maps and uncertainty maps"""
    st.subheader(f"XAI Analysis for {sample_name}")

    # Create columns for different visualizations
    cols = st.columns(2)

    with cols[0]:
        st.markdown("**Integrated Gradients Attribution Map**")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(xai_data['ig_map'], cmap='RdBu_r')
        ax.set_title("Integrated Gradients Attribution")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)
        plt.close()

    with cols[1]:
        st.markdown("**Uncertainty Map**")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(xai_data['uncertainty_map'], cmap='viridis')
        ax.set_title("Prediction Uncertainty")
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
        st.pyplot(fig)
        plt.close()

    # Ground truth and prediction comparison
    st.markdown("**Ground Truth vs Prediction**")
    cols = st.columns(2)

    with cols[0]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(xai_data['ground_truth'], cmap='gray')
        ax.set_title("Ground Truth")
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    with cols[1]:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(xai_data['prediction'], cmap='gray')
        ax.set_title("Prediction")
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    # Interactive point selection for cumulative histogram
    st.markdown("**Interactive Attribution Analysis**")
    st.info("Click on a point in the attribution map to see cumulative weighted attribution histogram")

    # For now, show histogram for center point
    center_point = (xai_data['ig_map'].shape[0] // 2, xai_data['ig_map'].shape[1] // 2)

    try:
        cumulative_hist, fig = calculate_cumulative_histogram(xai_data['ig_map'], center_point)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error calculating cumulative histogram: {e}")


def main():
    st.title("🫁 Lung Segmentation XAI Analysis Dashboard")
    st.markdown("---")

    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError:
        st.error("Config file not found. Please ensure config.yaml exists in the project directory.")
        return

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose Analysis Type:",
        ["Model Overview", "Training Analysis", "Model Comparison", "Sample Analysis"]
    )

    # Load available experiments
    experiments = get_available_experiments(config)

    if not experiments:
        st.error("No experiments found. Please run training first.")
        return

    if page == "Model Overview":
        st.header("Model Overview")

        # Display available experiments
        st.subheader("Available Experiments")

        for exp in experiments:
            with st.expander(f"📊 {exp['name']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Path:** {exp['path']}")
                    st.write(f"**Training Log:** {'✅' if exp['has_training_log'] else '❌'}")
                    st.write(f"**Evaluation:** {'✅' if exp['has_evaluation'] else '❌'}")

                with col2:
                    st.write(f"**Available Splits:** {', '.join(exp['splits_available'])}")

                    # Show basic metrics if available
                    if exp['splits_available']:
                        for split in exp['splits_available']:
                            eval_results = load_evaluation_results(exp['path'], split)
                            if eval_results:
                                st.metric(
                                    f"{split.title()} Dice Score",
                                    f"{eval_results['average_dice_score']:.4f}"
                                )

    elif page == "Training Analysis":
        st.header("Training Analysis")

        # Select experiments to analyze
        exp_names = [exp['name'] for exp in experiments if exp['has_training_log']]
        selected_exps = st.multiselect("Select Experiments to Analyze:", exp_names)

        if selected_exps:
            # Plot training curves for selected experiments
            fig = make_subplots(
                rows=len(selected_exps), cols=1,
                subplot_titles=selected_exps,
                vertical_spacing=0.1
            )

            for i, exp_name in enumerate(selected_exps):
                exp = next(exp for exp in experiments if exp['name'] == exp_name)
                training_df = load_training_history(exp['path'])

                if training_df is not None:
                    # Training loss
                    fig.add_trace(
                        go.Scatter(
                            x=training_df['epoch'],
                            y=training_df['train_loss'],
                            mode='lines+markers',
                            name=f'{exp_name} - Train',
                            line=dict(color='blue', width=2),
                            legendgroup=exp_name,
                            showlegend=True if i == 0 else False
                        ),
                        row=i + 1, col=1
                    )

                    # Validation loss
                    fig.add_trace(
                        go.Scatter(
                            x=training_df['epoch'],
                            y=training_df['val_loss'],
                            mode='lines+markers',
                            name=f'{exp_name} - Val',
                            line=dict(color='red', width=2),
                            legendgroup=exp_name,
                            showlegend=True if i == 0 else False
                        ),
                        row=i + 1, col=1
                    )

            fig.update_layout(height=300 * len(selected_exps))
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Comparison":
        st.header("Model Comparison")

        # Categorize models
        categorized_models = categorize_models(experiments, config)

        # Display comparison
        display_model_comparison(categorized_models, config)

    elif page == "Sample Analysis":
        st.header("Sample Analysis")

        # Display sample analysis
        display_sample_analysis(experiments, config)

    # Footer
    st.markdown("---")
    st.markdown("*Lung Segmentation XAI Analysis Dashboard - Built with Streamlit*")


if __name__ == "__main__":
    main()