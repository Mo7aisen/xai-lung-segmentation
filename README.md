# 🫁 XAI Lung Segmentation Analysis

> **Explainable AI for Medical Image Segmentation with Interactive Analysis Dashboard**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-ff6b6b)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Overview

This project implements a comprehensive **Explainable AI (XAI) framework** for lung segmentation in chest X-ray images, featuring an interactive analysis dashboard for comparative model evaluation. The system combines state-of-the-art deep learning models with advanced attribution methods to provide interpretable insights into model decision-making processes.

### 🎯 Key Features

- **🔬 Interactive XAI Dashboard**: Professional Streamlit-based interface for comparative analysis
- **🧠 Multiple Model Support**: U-Net architectures with various training configurations
- **📊 Advanced Attribution Methods**: Integrated Gradients with uncertainty quantification
- **📈 Comparative Analysis**: Side-by-side evaluation of different model states
- **🎚️ Real-time Visualization**: Dynamic threshold adjustment and overlay visualization
- **📤 Export Capabilities**: Comprehensive result export for research workflows
- **🔄 Version Control**: Professional Git workflow with organized project structure

## 🏗️ Project Structure

```
xai/
├── 📁 src/                          # Source code
│   ├── 📁 models/                   # Model architectures
│   │   └── model.py                 # U-Net implementation
│   ├── 📁 data/                     # Data handling
│   │   └── data_loader.py           # Dataset loaders
│   ├── 📁 training/                 # Training scripts
│   │   ├── train.py                 # Basic training
│   │   └── train_extended.py        # Extended training pipeline
│   ├── 📁 evaluation/               # Evaluation modules
│   │   └── evaluate.py              # Model evaluation
│   ├── 📁 visualization/            # Interactive dashboards
│   │   ├── app.py                   # Basic dashboard
│   │   └── app_enhanced.py          # Enhanced XAI dashboard
│   └── 📁 utils/                    # Utility functions
│       ├── utils.py                 # General utilities
│       ├── validation.py            # Validation functions
│       ├── generate_split_manifest.py
│       └── organize_model_checkpoints.py
├── 📁 scripts/                      # Automation scripts
│   ├── master_pipeline.sh           # Main pipeline
│   ├── continue_pipeline.sh         # Resume training
│   ├── run_training.sh              # Training execution
│   ├── run_evaluation.sh            # Evaluation execution
│   ├── run_extended_training.sh     # Extended training
│   ├── run_extended_evaluation.sh   # Extended evaluation
│   ├── monitor_progress.sh          # Progress monitoring
│   ├── setup_project.sh             # Project setup
│   └── legacy_scripts.py            # Legacy utilities
├── 📁 config/                       # Configuration files
│   ├── config.yaml                  # Main configuration
│   └── config_extended.yaml         # Extended config
├── 📁 outputs/                      # Training outputs
│   ├── 📁 model_checkpoints/        # Saved models
│   ├── 📁 evaluation_results/       # Evaluation data
│   └── 📁 xai_maps/                 # Attribution maps
├── 📁 exports/                      # Exported analysis results
├── 📁 tests/                        # Unit tests
├── 📁 docs/                         # Documentation
├── 📁 assets/                       # Static assets
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **CUDA-capable GPU** (recommended)
- **Git** for version control

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/xai-lung-segmentation.git
   cd xai-lung-segmentation
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure datasets**
   ```bash
   cp config/config.template.yaml config/config.yaml
   # Edit config/config.yaml with your dataset paths
   ```

### 🎮 Usage

#### 1. **Interactive XAI Dashboard**
Launch the enhanced analysis dashboard:
```bash
streamlit run src/visualization/app_enhanced.py
```

**Features:**
- **Three-column comparative analysis**
- **Real-time segmentation threshold adjustment**
- **Interactive pixel selection with coordinate controls**
- **Professional attribution visualizations**
- **Export functionality for research workflows**

#### 2. **Training Pipeline**
Run the complete training pipeline:
```bash
bash scripts/master_pipeline.sh
```

#### 3. **Model Evaluation**
Evaluate trained models:
```bash
bash scripts/run_extended_evaluation.sh
```

## 📊 Datasets

The project supports multiple chest X-ray datasets:

- **🏥 JSRT Dataset**: Japanese Society of Radiological Technology
- **🏥 Montgomery Dataset**: Montgomery County chest X-ray dataset
- **🔧 Custom datasets**: Configurable through YAML files

### Dataset Configuration

Edit `config/config.yaml`:
```yaml
datasets:
  jsrt:
    path: "/path/to/jsrt"
    images: "images"
    masks: "masks"
  montgomery:
    path: "/path/to/montgomery"
    images: "CXR_png"
    masks: "ManualMask"
```

## 🧠 Model Architecture

### U-Net Implementation
- **Encoder**: Progressive downsampling with skip connections
- **Decoder**: Progressive upsampling with concatenation
- **Output**: Sigmoid activation for binary segmentation

### Training Configurations
- **Full Dataset**: Complete training data
- **Half Dataset**: 50% of training data for comparison
- **Multiple Epochs**: 50, 150, 250+ epoch configurations

## 📈 XAI Methods

### Integrated Gradients
- **Attribution computation** for pixel-level explanations
- **Baseline integration** with multiple baselines
- **Uncertainty quantification** via Monte Carlo dropout

### Visualization Features
- **Heatmap overlays** with customizable colormaps
- **Cumulative attribution analysis** with distance-based weighting
- **Professional histogram displays** for distribution analysis

## 🔧 Configuration

### Environment Variables
```bash
export OUTPUT_DIR="/path/to/outputs"
export DATA_PATH="/path/to/datasets"
export MONTGOMERY_PATH="/path/to/montgomery"
export JSRT_PATH="/path/to/jsrt"
```

### YAML Configuration
See `config/config.yaml` for detailed configuration options.

## 📤 Export & Analysis

The dashboard supports comprehensive result export:

- **📊 Metrics**: CSV format with quantitative results
- **🖼️ Visualizations**: PNG format for figures
- **💾 Raw Data**: NPY/NPZ format for attribution maps
- **📋 Comparison Reports**: JSON format for multi-model analysis

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

## 📝 Documentation

Additional documentation available in the `docs/` directory:

- **🏗️ Architecture Guide**: Detailed model architecture
- **🔬 XAI Methods**: Explanation of attribution techniques
- **🎨 UI Guide**: Dashboard usage instructions
- **🚀 Deployment**: Production deployment guide

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{xai_lung_segmentation_2024,
  title={XAI Lung Segmentation Analysis: Explainable AI for Medical Image Segmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/xai-lung-segmentation}
}
```

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Streamlit Team** for the interactive dashboard framework
- **Medical Imaging Community** for dataset contributions
- **XAI Research Community** for attribution method development

## 📧 Contact

- **Author**: Your Name
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

<p align="center">
  <strong>🚀 Building Explainable AI for Medical Imaging 🚀</strong>
</p>