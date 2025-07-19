#!/bin/bash
# Master pipeline script - Complete rebuild from scratch
# SSH-safe with nohup, progress monitoring, and result overwriting

set -e

# Script configuration
SCRIPT_NAME="XAI_MASTER_PIPELINE"
LOG_DIR="pipeline_logs"
MAIN_LOG="$LOG_DIR/master_pipeline_$(date +%Y%m%d_%H%M%S).log"
PROGRESS_FILE="$LOG_DIR/progress.txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_and_echo() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "$message" | tee -a "$MAIN_LOG"
    echo "[$timestamp] $message" >> "$MAIN_LOG"
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
    echo "STEP_$step: $description" > "$PROGRESS_FILE"
}

print_status() {
    log_and_echo "${GREEN}✓${NC} $1"
}

print_warning() {
    log_and_echo "${YELLOW}⚠${NC} $1"
}

print_error() {
    log_and_echo "${RED}✗${NC} $1"
}

print_info() {
    log_and_echo "${PURPLE}ℹ${NC} $1"
}

# Function to run command with logging and error handling
run_with_logging() {
    local cmd="$1"
    local description="$2"
    local log_file="$LOG_DIR/${description// /_}.log"

    print_info "Running: $description"
    print_info "Command: $cmd"
    print_info "Log file: $log_file"

    if eval "$cmd" > "$log_file" 2>&1; then
        print_status "COMPLETED: $description"
        return 0
    else
        print_error "FAILED: $description"
        print_error "Check log: $log_file"
        print_error "Last 10 lines of log:"
        tail -n 10 "$log_file" | while read line; do
            log_and_echo "  $line"
        done
        return 1
    fi
}

# Function to clean old results
clean_old_results() {
    print_step "CLEANUP" "Removing old results to start fresh"

    # Remove outputs directory completely
    if [ -d "outputs" ]; then
        print_info "Removing old outputs directory..."
        rm -rf outputs
        print_status "Old outputs removed"
    fi

    # Remove any existing pipeline logs except current session
    if [ -d "$LOG_DIR" ]; then
        find "$LOG_DIR" -name "*.log" -not -name "$(basename $MAIN_LOG)" -mtime +1 -delete 2>/dev/null || true
    fi

    # Create fresh directory structure
    mkdir -p outputs/{logs,organized_models,extended_training_logs}
    mkdir -p "$LOG_DIR"

    print_status "Clean environment prepared"
}

# Function to check prerequisites
check_prerequisites() {
    print_step "PREREQ" "Checking prerequisites and environment"

    # Check conda environment
    if [ -z "$CONDA_DEFAULT_ENV" ]; then
        print_error "No conda environment active!"
        print_error "Please run: conda activate your_env_name"
        exit 1
    fi
    print_status "Conda environment: $CONDA_DEFAULT_ENV"

    # Check CUDA
    if command -v nvidia-smi >/dev/null 2>&1; then
        print_status "GPU available: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    else
        print_warning "No GPU detected - will use CPU (slower)"
    fi

    # Check required files
    required_files=("config.yaml" "train.py" "evaluate.py" "app.py" "model.py" "data_loader.py" "utils.py")
    for file in "${required_files[@]}"; do
        if [ -f "$file" ]; then
            print_status "$file found"
        else
            print_error "$file missing!"
            exit 1
        fi
    done

    # Check Python packages
    print_info "Checking Python packages..."
    python -c "
import torch, torchvision, numpy, pandas, matplotlib, seaborn, tqdm, PIL, yaml, streamlit
print('✓ All packages available')
" || {
        print_error "Missing Python packages!"
        exit 1
    }
    print_status "All Python packages available"
}

# Function to show pipeline plan
show_pipeline_plan() {
    print_step "PLAN" "Showing complete pipeline execution plan"

    log_and_echo ""
    log_and_echo "🎯 COMPLETE PIPELINE EXECUTION PLAN:"
    log_and_echo ""
    log_and_echo "Phase 1: Initial Training (8 experiments)"
    log_and_echo "  • Montgomery Full: 50, 150 epochs"
    log_and_echo "  • Montgomery Half: 50, 150 epochs"
    log_and_echo "  • JSRT Full: 50, 150 epochs"
    log_and_echo "  • JSRT Half: 50, 150 epochs"
    log_and_echo ""
    log_and_echo "Phase 2: Extended Training (3 experiments)"
    log_and_echo "  • Montgomery Full → 250 epochs (overfitting)"
    log_and_echo "  • Montgomery Half → 300 epochs (overfitting)"
    log_and_echo "  • JSRT Half → 250 epochs (overfitting)"
    log_and_echo ""
    log_and_echo "Phase 3: Comprehensive Evaluation"
    log_and_echo "  • All models × 3 states × 3 splits = 99 evaluations"
    log_and_echo "  • Using supervisor's exact epoch specifications"
    log_and_echo ""
    log_and_echo "Phase 4: Validation & Dashboard"
    log_and_echo "  • Validate all supervisor requirements met"
    log_and_echo "  • Prepare for Streamlit dashboard"
    log_and_echo ""
    log_and_echo "📊 Supervisor's Epoch Specifications:"
    log_and_echo "  Montgomery Full: 5→75,105→250"
    log_and_echo "  Montgomery Half: 10→115,140→300"
    log_and_echo "  JSRT Full: 5→35→150"
    log_and_echo "  JSRT Half: 5→70→250"
    log_and_echo ""
}

# Main pipeline execution
main_pipeline() {
    print_header "XAI LUNG SEGMENTATION - COMPLETE PIPELINE REBUILD"

    log_and_echo "🚀 Starting complete pipeline execution..."
    log_and_echo "📅 Started at: $(date)"
    log_and_echo "🖥️  Running on: $(hostname)"
    log_and_echo "📂 Working directory: $(pwd)"
    log_and_echo "📝 Main log: $MAIN_LOG"
    log_and_echo ""

    # Phase 0: Setup
    check_prerequisites
    show_pipeline_plan
    clean_old_results

    # Phase 1: Initial Training
    print_step "TRAIN1" "Phase 1: Initial Training (50 & 150 epochs)"
    if ! run_with_logging "./run_training.sh" "initial_training"; then
        print_error "Initial training failed!"
        exit 1
    fi

    # Phase 2: Extended Training
    print_step "TRAIN2" "Phase 2: Extended Training (250 & 300 epochs)"
    if ! run_with_logging "./run_extended_training.sh" "extended_training"; then
        print_error "Extended training failed!"
        exit 1
    fi

    # Phase 3: Standard Evaluation
    print_step "EVAL1" "Phase 3a: Standard Model Evaluation"
    if ! run_with_logging "./run_evaluation.sh" "standard_evaluation"; then
        print_error "Standard evaluation failed!"
        exit 1
    fi

    # Phase 4: Extended Evaluation
    print_step "EVAL2" "Phase 3b: Extended Model Evaluation"
    if ! run_with_logging "./run_extended_evaluation.sh" "extended_evaluation"; then
        print_error "Extended evaluation failed!"
        exit 1
    fi

    # Phase 5: Validation
    print_step "VALIDATE" "Phase 4: Validation of Supervisor Requirements"
    if [ -f "validate_supervisor_requirements.py" ]; then
        if ! run_with_logging "python validate_supervisor_requirements.py" "supervisor_validation"; then
            print_warning "Supervisor validation had issues (check log)"
        fi
    else
        print_warning "Supervisor validation script not found"
    fi

    # Phase 6: Final Summary
    print_step "SUMMARY" "Phase 5: Final Summary and Status"
    generate_completion_report
}

# Function to generate completion report
generate_completion_report() {
    local report_file="$LOG_DIR/completion_report.txt"
    local end_time=$(date)

    {
        echo "XAI LUNG SEGMENTATION - PIPELINE COMPLETION REPORT"
        echo "=================================================="
        echo ""
        echo "Execution Summary:"
        echo "  Start Time: $(head -n 20 "$MAIN_LOG" | grep "Started at" | cut -d: -f2-)"
        echo "  End Time: $end_time"
        echo "  Total Duration: $(( ($(date +%s) - $(stat -c %Y "$MAIN_LOG")) / 60 )) minutes"
        echo ""
        echo "Results Location:"
        echo "  Main Log: $MAIN_LOG"
        echo "  Outputs: $(pwd)/outputs/"
        echo "  Models: $(find outputs -name "*.pth" | wc -l) model files"
        echo "  Evaluations: $(find outputs -name "_evaluation_summary.json" | wc -l) evaluation summaries"
        echo ""
        echo "Available Models:"
        find outputs -maxdepth 1 -type d -name "unet_*" | sort | while read dir; do
            echo "  ✓ $(basename "$dir")"
        done
        echo ""
        echo "Next Steps:"
        echo "  1. Launch dashboard: streamlit run app.py"
        echo "  2. Access at: http://localhost:8501"
        echo "  3. Compare models across fitting states"
        echo ""
        echo "Supervisor's Epoch Analysis Ready:"
        echo "  ✓ Underfitting models (epochs 5-10)"
        echo "  ✓ Good fitting models (epochs 35-140)"
        echo "  ✓ Overfitting models (epochs 150-300)"
        echo ""
    } > "$report_file"

    # Display the report
    cat "$report_file"
    log_and_echo ""
    print_status "Completion report saved: $report_file"
}

# Function to monitor progress (for SSH monitoring)
create_progress_monitor() {
    cat > monitor_progress.sh << 'EOF'
#!/bin/bash
# Progress monitoring script
LOG_DIR="pipeline_logs"
PROGRESS_FILE="$LOG_DIR/progress.txt"

echo "XAI Pipeline Progress Monitor"
echo "============================="
echo ""

if [ ! -f "$PROGRESS_FILE" ]; then
    echo "❌ Pipeline not running or progress file not found"
    exit 1
fi

while true; do
    clear
    echo "XAI Pipeline Progress Monitor - $(date)"
    echo "============================="
    echo ""

    if [ -f "$PROGRESS_FILE" ]; then
        echo "📊 Current Step:"
        cat "$PROGRESS_FILE"
        echo ""
    fi

    echo "📁 Recent Activity:"
    if [ -d "$LOG_DIR" ]; then
        ls -lt "$LOG_DIR"/*.log 2>/dev/null | head -5 | while read line; do
            echo "  $line"
        done
    fi

    echo ""
    echo "🔄 GPU Status:"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    else
        echo "  No GPU detected"
    fi

    echo ""
    echo "Press Ctrl+C to exit monitor"
    sleep 10
done
EOF
    chmod +x monitor_progress.sh
    print_status "Progress monitor created: ./monitor_progress.sh"
}

# Trap to handle script interruption
cleanup_on_exit() {
    local exit_code=$?
    echo ""
    if [ $exit_code -eq 0 ]; then
        print_header "PIPELINE COMPLETED SUCCESSFULLY! 🎉"
    else
        print_header "PIPELINE INTERRUPTED OR FAILED ❌"
        print_error "Exit code: $exit_code"
        print_info "Check logs in: $LOG_DIR/"
    fi

    echo "FINISHED: $(date)" >> "$PROGRESS_FILE"
}

trap cleanup_on_exit EXIT

# Pre-execution checks
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: Must run from project root directory (config.yaml not found)"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Create progress monitor
create_progress_monitor

# Main execution
log_and_echo "🚀 STARTING COMPLETE XAI PIPELINE REBUILD"
log_and_echo "📝 This will run EVERYTHING from scratch with supervisor's specifications"
log_and_echo "🔒 SSH-safe execution with nohup"
log_and_echo ""

# Show instructions for SSH usage
if [ -t 0 ]; then
    echo "🔗 SSH Usage Instructions:"
    echo "  To run safely in background:"
    echo "    nohup ./master_pipeline.sh > pipeline_output.log 2>&1 &"
    echo ""
    echo "  To monitor progress:"
    echo "    tail -f pipeline_output.log"
    echo "    # or"
    echo "    ./monitor_progress.sh"
    echo ""
    echo "  Current execution will start in 5 seconds..."
    echo "  Press Ctrl+C to cancel and run with nohup instead"
    sleep 5
fi

# Execute main pipeline
main_pipeline