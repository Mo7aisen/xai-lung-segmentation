#!/bin/bash
# Continue pipeline from where extended evaluation failed
# Keeps all existing work and continues from the failure point

set -e

# Script configuration
SCRIPT_NAME="XAI_CONTINUE_PIPELINE"
LOG_DIR="pipeline_logs"
CONTINUE_LOG="$LOG_DIR/continue_pipeline_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="$LOG_DIR/progress.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_and_echo() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "$message" | tee -a "$CONTINUE_LOG"
    echo "[$timestamp] $message" >> "$CONTINUE_LOG"
}

print_header() {
    local title="$1"
    local line="========================================================"
    log_and_echo "${CYAN}$line${NC}"
    log_and_echo "${CYAN}    $title${NC}"
    log_and_echo "${CYAN}$line${NC}"
}

print_step() {
    local step="$1"
    local description="$2"
    log_and_echo "${BLUE}[$step]${NC} $description"
    echo "CONTINUE_$step: $description" > "$PROGRESS_FILE"
}

print_status() { log_and_echo "${GREEN}✓${NC} $1"; }
print_warning() { log_and_echo "${YELLOW}⚠${NC} $1"; }
print_error() { log_and_echo "${RED}✗${NC} $1"; }
print_info() { log_and_echo "${PURPLE}ℹ${NC} $1"; }

# Function to run command with logging
run_with_logging() {
    local cmd="$1"
    local description="$2"
    local log_file="$LOG_DIR/continue_${description// /_}.log"

    print_info "Running: $description"
    print_info "Command: $cmd"
    print_info "Log file: $log_file"

    if eval "$cmd" > "$log_file" 2>&1; then
        print_status "COMPLETED: $description"
        return 0
    else
        print_error "FAILED: $description"
        print_error "Check log: $log_file"
        print_error "Last 10 lines:"
        tail -n 10 "$log_file" | while read line; do
            log_and_echo "  $line"
        done
        return 1
    fi
}

# Function to check current status
check_current_status() {
    print_step "STATUS" "Checking current pipeline status"

    # Check training completion
    local training_models=(
        "unet_montgomery_full_150"
        "unet_montgomery_full_50"
        "unet_montgomery_half_150"
        "unet_montgomery_half_50"
        "unet_jsrt_full_150"
        "unet_jsrt_full_50"
        "unet_jsrt_half_150"
        "unet_jsrt_half_50"
    )

    local extended_models=(
        "unet_montgomery_full_250"
        "unet_montgomery_half_300"
        "unet_jsrt_half_250"
    )

    log_and_echo ""
    log_and_echo "📊 CURRENT STATUS SUMMARY:"
    log_and_echo ""

    # Check initial training
    log_and_echo "🚀 Initial Training Status:"
    local initial_complete=0
    for model in "${training_models[@]}"; do
        if [ -f "outputs/$model/final_model.pth" ]; then
            print_status "$model"
            initial_complete=$((initial_complete + 1))
        else
            print_error "$model (missing)"
        fi
    done
    log_and_echo "  Complete: $initial_complete/${#training_models[@]}"

    # Check extended training
    log_and_echo ""
    log_and_echo "📈 Extended Training Status:"
    local extended_complete=0
    for model in "${extended_models[@]}"; do
        if [ -f "outputs/$model/final_model.pth" ]; then
            print_status "$model"
            extended_complete=$((extended_complete + 1))
        else
            print_error "$model (missing)"
        fi
    done
    log_and_echo "  Complete: $extended_complete/${#extended_models[@]}"

    # Check evaluations
    log_and_echo ""
    log_and_echo "🔬 Evaluation Status:"

    # Count standard evaluations
    local std_eval_count=0
    for model in "${training_models[@]}"; do
        for state in "underfitting" "good_fitting" "overfitting"; do
            for split in "test" "validation" "training"; do
                if [ -f "outputs/$model/evaluation/$state/$split/_evaluation_summary.json" ]; then
                    std_eval_count=$((std_eval_count + 1))
                fi
            done
        done
    done
    log_and_echo "  Standard evaluations: $std_eval_count/$((${#training_models[@]} * 3 * 3))"

    # Count extended evaluations
    local ext_eval_count=0
    for model in "${extended_models[@]}"; do
        for state in "underfitting" "good_fitting" "overfitting"; do
            for split in "test" "validation" "training"; do
                if [ -f "outputs/$model/evaluation/$state/$split/_evaluation_summary.json" ]; then
                    ext_eval_count=$((ext_eval_count + 1))
                fi
            done
        done
    done
    log_and_echo "  Extended evaluations: $ext_eval_count/$((${#extended_models[@]} * 3 * 3))"

    log_and_echo ""

    # Determine what needs to be done
    if [ $initial_complete -lt ${#training_models[@]} ]; then
        log_and_echo "❌ Initial training incomplete - need to run training first"
        return 1
    elif [ $extended_complete -lt ${#extended_models[@]} ]; then
        log_and_echo "❌ Extended training incomplete - need to run extended training"
        return 2
    elif [ $ext_eval_count -lt $((${#extended_models[@]} * 3 * 3)) ]; then
        log_and_echo "⚠️  Extended evaluation incomplete - this is where we'll continue"
        return 0
    else
        log_and_echo "✅ Everything appears complete!"
        return 0
    fi
}

# Function to continue from extended evaluation
continue_from_extended_eval() {
    print_step "CONTINUE" "Continuing from extended evaluation failure point"

    # First, replace the problematic script
    print_info "Updating extended evaluation script..."

    # Check if we have the fixed script
    if ! grep -q "No Interactive Input" run_extended_evaluation.sh 2>/dev/null; then
        print_warning "Using current run_extended_evaluation.sh (may need manual update)"
    fi

    # Run the fixed extended evaluation
    print_step "EVAL_EXT" "Running fixed extended evaluation"
    if ! run_with_logging "./run_extended_evaluation.sh" "fixed_extended_evaluation"; then
        print_error "Extended evaluation still failing!"
        print_info "Trying manual evaluation approach..."

        # Manual evaluation as fallback
        manual_extended_evaluation
    fi
}

# Function to run manual extended evaluation
manual_extended_evaluation() {
    print_step "MANUAL_EVAL" "Running manual extended evaluation"

    local extended_models=("unet_montgomery_full_250" "unet_montgomery_half_300" "unet_jsrt_half_250")
    local states=("underfitting" "good_fitting" "overfitting")
    local splits=("test" "validation" "training")

    for model in "${extended_models[@]}"; do
        if [ ! -f "outputs/$model/final_model.pth" ]; then
            print_warning "Skipping $model (not trained)"
            continue
        fi

        print_info "Evaluating $model..."

        for state in "${states[@]}"; do
            for split in "${splits[@]}"; do
                local summary_file="outputs/$model/evaluation/$state/$split/_evaluation_summary.json"

                if [ -f "$summary_file" ]; then
                    print_status "Already done: $model/$state/$split"
                    continue
                fi

                print_info "Running: $model/$state/$split"
                local log_file="$LOG_DIR/manual_eval_${model}_${state}_${split}.log"

                if python3 evaluate.py \
                    --run_name "$model" \
                    --state "$state" \
                    --split "$split" \
                    > "$log_file" 2>&1; then
                    print_status "✓ $model/$state/$split"
                else
                    print_error "✗ $model/$state/$split (check $log_file)"
                fi
            done
        done
    done
}

# Function to complete remaining pipeline steps
complete_pipeline() {
    print_step "VALIDATE" "Running final validation"

    # Run supervisor validation if available
    if [ -f "validate_supervisor_requirements.py" ]; then
        if ! run_with_logging "python validate_supervisor_requirements.py" "supervisor_validation"; then
            print_warning "Supervisor validation had issues (check log)"
        fi
    fi

    # Generate final report
    print_step "REPORT" "Generating completion report"
    generate_final_report
}

# Function to generate final report
generate_final_report() {
    local report_file="$LOG_DIR/continue_completion_report.txt"
    local end_time=$(date)

    {
        echo "XAI PIPELINE - CONTINUATION COMPLETION REPORT"
        echo "============================================="
        echo ""
        echo "Continuation Summary:"
        echo "  Restart Time: $(head -n 5 "$CONTINUE_LOG" | grep "CONTINUE PIPELINE" -A 1 | tail -1)"
        echo "  End Time: $end_time"
        echo ""
        echo "Status Check Results:"
        echo "  Models Available:"
        find outputs -maxdepth 1 -type d -name "unet_*" | wc -l
        echo "  Evaluation Summaries:"
        find outputs -name "_evaluation_summary.json" | wc -l
        echo ""
        echo "Available Models:"
        find outputs -maxdepth 1 -type d -name "unet_*" | sort | while read dir; do
            echo "  ✓ $(basename "$dir")"
        done
        echo ""
        echo "Extended Models Status:"
        for model in "unet_montgomery_full_250" "unet_montgomery_half_300" "unet_jsrt_half_250"; do
            if [ -f "outputs/$model/final_model.pth" ]; then
                echo "  ✓ $model (trained)"
                # Count evaluations
                eval_count=$(find "outputs/$model/evaluation" -name "_evaluation_summary.json" 2>/dev/null | wc -l)
                echo "    Evaluations: $eval_count/9"
            else
                echo "  ✗ $model (not trained)"
            fi
        done
        echo ""
        echo "Next Steps:"
        echo "  1. Check evaluation completeness above"
        echo "  2. Launch dashboard: streamlit run app.py"
        echo "  3. Analyze results with supervisor's specifications"
        echo ""
    } > "$report_file"

    cat "$report_file"
    print_status "Report saved: $report_file"
}

# Main execution
main() {
    print_header "CONTINUE XAI PIPELINE FROM FAILURE POINT"

    log_and_echo "🔄 Continuing XAI pipeline execution..."
    log_and_echo "📅 Restart time: $(date)"
    log_and_echo "📝 Continue log: $CONTINUE_LOG"
    log_and_echo ""

    # Check current status
    check_current_status
    local status_code=$?

    case $status_code in
        1)
            print_error "Initial training incomplete - please run: ./run_training.sh"
            exit 1
            ;;
        2)
            print_error "Extended training incomplete - please run: ./run_extended_training.sh"
            exit 1
            ;;
        0)
            print_info "Continuing from extended evaluation..."
            continue_from_extended_eval
            complete_pipeline
            ;;
    esac

    print_header "PIPELINE CONTINUATION COMPLETED! 🎉"
}

# Create log directory
mkdir -p "$LOG_DIR"

# Check prerequisites
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: Must run from project root directory"
    exit 1
fi

# Execute main function
main "$@"