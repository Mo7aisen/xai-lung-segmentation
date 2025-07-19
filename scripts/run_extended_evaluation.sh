#!/bin/bash
# Fixed evaluation script for extended training models
# Removes interactive input that causes issues with nohup

set -e

echo "========================================================"
echo "    EVALUATING EXTENDED TRAINING MODELS                 "
echo "========================================================"

# Extended models to evaluate
declare -a extended_models=(
    "unet_montgomery_full_250"
    "unet_montgomery_half_300"
    "unet_jsrt_half_250"
)

# Define states for evaluation
declare -a states=("underfitting" "good_fitting" "overfitting")
declare -a splits=("test" "validation" "training")

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

# Function to check if model directory exists
check_model_exists() {
    local run_name=$1
    local model_dir="outputs/${run_name}"

    if [ ! -d "$model_dir" ]; then
        print_error "Model directory not found: $model_dir"
        print_info "Hint: Run extended training first with: ./run_extended_training.sh"
        return 1
    fi

    # Check if final model exists
    if [ ! -f "$model_dir/final_model.pth" ]; then
        print_error "Final model not found: $model_dir/final_model.pth"
        return 1
    fi

    return 0
}

# Function to ensure directory exists
ensure_directory() {
    local dir_path=$1
    if [ ! -d "$dir_path" ]; then
        mkdir -p "$dir_path"
        print_info "Created directory: $dir_path"
    fi
}

# Function to check if evaluation already exists
evaluation_exists() {
    local run_name=$1
    local state=$2
    local split=$3
    local summary_file="outputs/${run_name}/evaluation/${state}/${split}/_evaluation_summary.json"

    if [ -f "$summary_file" ]; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to run single evaluation
run_single_evaluation() {
    local run_name=$1
    local state=$2
    local split=$3

    print_info "Evaluating: $run_name | $state | $split"

    if evaluation_exists "$run_name" "$state" "$split"; then
        print_status "Already completed: $run_name/$state/$split"
        return 0
    fi

    # Ensure log directory exists
    local log_dir="outputs/${run_name}"
    ensure_directory "$log_dir"

    local log_file="outputs/${run_name}/eval_${state}_${split}_extended.log"

    # Run evaluation with proper error handling
    if python3 evaluate.py \
        --run_name "$run_name" \
        --state "$state" \
        --split "$split" \
        > "$log_file" 2>&1; then

        print_status "✓ Completed: $run_name/$state/$split"
        return 0
    else
        print_error "✗ Failed: $run_name/$state/$split"
        print_error "Check log: $log_file"

        # Show last few lines for debugging
        if [ -f "$log_file" ]; then
            echo "Last 5 lines of error log:"
            tail -n 5 "$log_file" | while read line; do
                echo "  $line"
            done
        fi
        return 1
    fi
}

# Main execution
main() {
    print_info "Starting extended model evaluation..."
    print_info "Models to evaluate: ${extended_models[*]}"
    print_info "States: ${states[*]}"
    print_info "Splits: ${splits[*]}"
    echo ""

    local total_evaluations=0
    local completed_evaluations=0
    local failed_evaluations=0
    local skipped_evaluations=0

    # Count total evaluations needed
    total_evaluations=$((${#extended_models[@]} * ${#states[@]} * ${#splits[@]}))
    print_info "Total evaluations needed: $total_evaluations"
    echo ""

    for run_name in "${extended_models[@]}"; do
        echo "========================================================"
        echo "Evaluating: $run_name"

        # Check if model exists
        if ! check_model_exists "$run_name"; then
            print_warning "Skipping $run_name (model not found)"
            skipped_evaluations=$((skipped_evaluations + ${#states[@]} * ${#splits[@]}))
            continue
        fi

        for state in "${states[@]}"; do
            print_info "State: $state"

            for split in "${splits[@]}"; do
                if evaluation_exists "$run_name" "$state" "$split"; then
                    print_status "Already exists: $split"
                    completed_evaluations=$((completed_evaluations + 1))
                else
                    print_info "Running: $split"

                    if run_single_evaluation "$run_name" "$state" "$split"; then
                        completed_evaluations=$((completed_evaluations + 1))
                    else
                        failed_evaluations=$((failed_evaluations + 1))
                        print_warning "Continuing with remaining evaluations..."
                    fi
                fi
            done
        done
    done

    echo ""
    echo "========================================================"
    echo "    EXTENDED EVALUATION SUMMARY                        "
    echo "========================================================"
    echo ""
    echo "📊 Evaluation Statistics:"
    echo "  Total needed: $total_evaluations"
    echo "  Completed: $completed_evaluations"
    echo "  Failed: $failed_evaluations"
    echo "  Skipped: $skipped_evaluations"
    echo ""

    if [ $failed_evaluations -eq 0 ] && [ $skipped_evaluations -eq 0 ]; then
        print_status "🎉 ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
    elif [ $failed_evaluations -eq 0 ]; then
        print_warning "⚠️ Some models were skipped (not trained yet)"
    else
        print_warning "⚠️ Some evaluations failed (check logs above)"
    fi

    echo ""
    echo "Summary of evaluated models:"
    for run_name in "${extended_models[@]}"; do
        if [ -d "outputs/${run_name}" ] && [ -f "outputs/${run_name}/final_model.pth" ]; then
            # Count completed evaluations for this model
            local model_evals=0
            for state in "${states[@]}"; do
                for split in "${splits[@]}"; do
                    if evaluation_exists "$run_name" "$state" "$split"; then
                        model_evals=$((model_evals + 1))
                    fi
                done
            done
            print_status "$run_name ($model_evals/9 evaluations)"
        else
            print_error "$run_name (model not found)"
        fi
    done

    echo ""
    echo "🎨 Next steps:"
    echo "  1. Check any failed evaluations in logs"
    echo "  2. Launch dashboard: streamlit run app.py"
    echo "  3. Analyze results across fitting states"

    # Return appropriate exit code
    if [ $failed_evaluations -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Check if we're in the right directory
if [ ! -f "config.yaml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Check if evaluate.py exists and has supervisor specifications
if [ ! -f "evaluate.py" ]; then
    print_error "evaluate.py not found!"
    exit 1
fi

if ! grep -q "get_supervisor_epoch_mapping" evaluate.py 2>/dev/null; then
    print_warning "evaluate.py may not have supervisor's epoch specifications"
    print_info "Make sure you're using the updated evaluate.py"
fi

# Run main function
main "$@"