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
