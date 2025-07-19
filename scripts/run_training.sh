#!/bin/bash
set -e
echo "========================================================"
echo "      STARTING XAI PROJECT - CONFIG-DRIVEN TRAINING     "
echo "========================================================"

# This script reads the config.yaml file to run all defined experiments.

experiments=$(python3 -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(' '.join(config['experiments'].keys()))")

for run_name in $experiments; do
    echo "--------------------------------------------------------"
    echo "Processing Training: $run_name"
    
    # The output directory is now defined in the config
    run_dir=$(python3 -c "import yaml; config = yaml.safe_load(open('config.yaml')); print(config['output_base_dir'])")/"$run_name"
    
    if [ -f "${run_dir}/final_model.pth" ]; then
        echo "Final model already exists. SKIPPING."
    else
        # --- FIXED: Create the run directory before attempting to write to it. ---
        mkdir -p "$run_dir"
        # The python script will read its own config, we just pass the run name
        # Redirect output to a log file inside the run's directory
        python3 train.py --run_name "$run_name" > "${run_dir}/train.log" 2>&1
    fi
done

echo "========================================================"
echo "      ALL TRAINING PROCESSES HAVE COMPLETED.            "
echo "========================================================"

