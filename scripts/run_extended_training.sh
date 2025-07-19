#!/bin/bash
# Extended training script based on supervisor's meeting notes
# Trains models to specific epochs for clear overfitting demonstration

set -e

echo "========================================================"
echo "    EXTENDED TRAINING - SUPERVISOR'S SPECIFICATIONS     "
echo "========================================================"
echo "Training models to demonstrate overfitting as discussed"
echo "in the meeting with supervisors Dr. Hadházi and Dr. Hullám"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Check if Phoenix server is available
if ! nvidia-smi &> /dev/null; then
    print_warning "GPU not available. Training will use CPU (much slower)."
    echo "Consider running on a machine with CUDA support."
fi

# Create extended training log directory
mkdir -p outputs/extended_training_logs

# Extended training jobs based on supervisor's meeting notes
declare -A training_jobs=(
    ["montgomery_full"]="250"    # Supervisor: overfitting at 250 epochs
    ["montgomery_half"]="300"    # Supervisor: overfitting at 300 epochs
    ["jsrt_half"]="250"          # Supervisor: overfitting at 250 epochs
)

echo "📋 Planned Extended Training (Supervisor's Specifications):"
echo ""
echo "Dataset               | Target Epochs | Purpose"
echo "----------------------|---------------|------------------"
echo "Montgomery Full       | 250           | Overfitting demo"
echo "Montgomery Half       | 300           | Overfitting demo"
echo "JSRT Half            | 250           | Overfitting demo"
echo ""
echo "📝 Note: JSRT Full already shows overfitting at 150 epochs"
echo ""

# Function to check prerequisites
check_prerequisites() {
    local dataset=$1
    local base_epochs

    case $dataset in
        "montgomery_full")
            base_epochs="150"
            ;;
        "montgomery_half")
            base_epochs="150"
            ;;
        "jsrt_half")
            base_epochs="150"
            ;;
    esac

    local base_run="unet_${dataset}_${base_epochs}"
    local base_model="outputs/${base_run}/final_model.pth"

    if [ ! -f "$base_model" ]; then
        print_error "Prerequisites missing for $dataset"
        echo "  Required: $base_model"
        echo "  Solution: Run ./run_training.sh first"
        return 1
    else
        print_status "Prerequisites met for $dataset"
        return 0
    fi
}

# Function to show supervisor's epoch mapping
show_supervisor_mapping() {
    echo ""
    print_info "Supervisor's Epoch Specifications for XAI Analysis:"
    echo ""
    echo "Montgomery Full Dataset:"
    echo "  • Underfitting: Epoch 5"
    echo "  • Good Fitting: Epochs 75, 105"
    echo "  • Overfitting: Epoch 250 ⭐ (This training)"
    echo ""
    echo "Montgomery Half Dataset:"
    echo "  • Underfitting: Epoch 10"
    echo "  • Good Fitting: Epochs 115, 140"
    echo "  • Overfitting: Epoch 300 ⭐ (This training)"
    echo ""
    echo "JSRT Half Dataset:"
    echo "  • Underfitting: Epoch 5"
    echo "  • Good Fitting: Epoch 70"
    echo "  • Overfitting: Epoch 250 ⭐ (This training)"
    echo ""
}

show_supervisor_mapping

# Check all prerequisites first
echo "Checking prerequisites..."
all_prereqs_met=true

for dataset in "${!training_jobs[@]}"; do
    if ! check_prerequisites "$dataset"; then
        all_prereqs_met=false
    fi
done

if [ "$all_prereqs_met" = false ]; then
    echo ""
    print_error "Some prerequisites are missing. Please run initial training first:"
    echo "  ./run_training.sh"
    exit 1
fi

echo ""
print_status "All prerequisites met. Starting extended training..."

# Run each extended training job
for dataset in "${!training_jobs[@]}"; do
    target_epochs="${training_jobs[$dataset]}"
    log_file="outputs/extended_training_logs/${dataset}_to_${target_epochs}_epochs.log"

    echo ""
    echo "========================================================"
    echo "🚀 TRAINING: $dataset to $target_epochs epochs"
    echo "========================================================"

    # Check if already completed
    final_model="outputs/unet_${dataset}_${target_epochs}/final_model.pth"
    if [ -f "$final_model" ]; then
        print_status "Already completed: $dataset to $target_epochs epochs"
        echo "  Model: $final_model"
        continue
    fi

    print_info "Starting extended training..."
    echo "  Dataset: $dataset"
    echo "  Target epochs: $target_epochs"
    echo "  Log file: $log_file"
    echo "  Purpose: Demonstrate overfitting (supervisor's specification)"

    # Run extended training
    start_time=$(date +%s)

    if python3 train_extended.py \
        --dataset "$dataset" \
        --target_epochs "$target_epochs" \
        > "$log_file" 2>&1; then

        end_time=$(date +%s)
        duration=$((end_time - start_time))
        duration_min=$((duration / 60))

        print_status "✅ COMPLETED: $dataset ($duration_min minutes)"
        echo "  Final model: outputs/unet_${dataset}_${target_epochs}/final_model.pth"
        echo "  Training log: $log_file"

        # Show some training info if available
        if [ -f "outputs/unet_${dataset}_${target_epochs}/training_log.csv" ]; then
            echo "  Training completed to epoch: $(tail -n 1 outputs/unet_${dataset}_${target_epochs}/training_log.csv | cut -d',' -f1)"
        fi

    else
        print_error "❌ FAILED: $dataset"
        echo "  Check log file for details: $log_file"

        # Show last few lines of log for quick debugging
        if [ -f "$log_file" ]; then
            echo ""
            echo "Last few lines of log:"
            tail -n 10 "$log_file"
        fi

        exit 1
    fi
done

echo ""
echo "========================================================"
echo "    🎉 EXTENDED TRAINING COMPLETED SUCCESSFULLY         "
echo "========================================================"
echo ""
echo "📊 Summary of Extended Models:"
for dataset in "${!training_jobs[@]}"; do
    target_epochs="${training_jobs[$dataset]}"
    model_path="outputs/unet_${dataset}_${target_epochs}/final_model.pth"

    if [ -f "$model_path" ]; then
        print_status "$dataset → $target_epochs epochs"
    else
        print_error "$dataset → $target_epochs epochs (MISSING)"
    fi
done

echo ""
echo "📋 Next Steps:"
echo ""
echo "1. 🔬 Run evaluation on extended models:"
echo "   ./run_extended_evaluation.sh"
echo ""
echo "2. 🎯 XAI analysis will now include supervisor's specified epochs:"
echo "   • Underfitting: Early epochs (5-10)"
echo "   • Good fitting: Supervisor-specified epochs (35-140)"
echo "   • Overfitting: Extended training epochs (250-300)"
echo ""
echo "3. 🎨 Launch comparative analysis:"
echo "   streamlit run app.py"
echo ""
echo "🎯 The extended models will demonstrate clear overfitting"
echo "   patterns as discussed in the supervisor meeting!"
echo ""

# Create a completion marker
echo "Extended training completed on $(date)" > outputs/extended_training_logs/completion_marker.txt

print_status "Extended training pipeline completed successfully! 🚀"