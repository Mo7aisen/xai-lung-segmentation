#!/bin/bash
set -e
echo "========================================================"
echo "      STARTING XAI PROJECT - FULL EVALUATION SUITE      "
echo "========================================================"

OUTPUT_BASE_DIR="./outputs"

declare -a model_runs=(
    "unet_montgomery_full_150"
    "unet_montgomery_full_50"
    "unet_montgomery_half_150"
    "unet_montgomery_half_50"
    "unet_jsrt_full_150"
    "unet_jsrt_full_50"
    "unet_jsrt_half_150"
    "unet_jsrt_half_50"
)
# --- NEW: Define the states and their corresponding model files ---
declare -A states=(
    ["underfitting"]="snapshots/epoch_10.pth"
    ["good_fitting"]="best_model.pth"
    ["overfitting"]="final_model.pth"
)

for run_name in "${model_runs[@]}"; do
    echo "========================================================"
    echo "Processing Evaluation for: $run_name"
    
    run_dir="${OUTPUT_BASE_DIR}/${run_name}"
    if [ ! -d "$run_dir" ]; then echo "Run directory not found. SKIPPING."; continue; fi

    # --- NEW: Loop through each defined state ---
    for state in "${!states[@]}"; do
        model_file="${states[$state]}"
        model_snapshot_path="${run_dir}/${model_file}"
        
        echo "--------------------------------------------------------"
        echo "Evaluating '$state' state"

        if [ ! -f "$model_snapshot_path" ]; then
            echo "Model for '$state' state not found at $model_snapshot_path. SKIPPING."
            continue
        fi

        # We evaluate on all three data splits for each state
        for split in "test" "validation" "training"; do
            summary_file="${run_dir}/evaluation/${state}/${split}/_evaluation_summary.json"
            if [ -f "$summary_file" ]; then
                echo "Evaluation for '$state' on '$split' split already exists. SKIPPING."
            else
                log_file="${run_dir}/eval_${state}_${split}.log"
                
                # --- FIXED: Passing the correct arguments to the updated evaluate.py script ---
                python3 evaluate.py \
                    --run_name "$run_name" \
                    --state "$state" \
                    --split "$split" 2>&1 | tee "$log_file"
            fi
        done
    done
done

echo "========================================================"
echo "      ALL EVALUATION PROCESSES HAVE COMPLETED.          "
echo "========================================================"
