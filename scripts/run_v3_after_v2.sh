#!/bin/bash
# Wait for a running v2 training process to finish, then start v3.
# Usage: ./scripts/run_v3_after_v2.sh [PID]

V2_PID="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$V2_PID" ]; then
    echo "Usage: $0 <v2-training-PID>"
    echo "  Pass the PID of the running v2 training process."
    exit 1
fi

echo "=== Tibeb v2→v3 Training Chain ==="
echo "Waiting for v2 training (PID $V2_PID) to finish..."

while kill -0 "$V2_PID" 2>/dev/null; do
    echo "  $(date '+%H:%M:%S') — v2 still running..."
    sleep 60
done

echo ""
echo "v2 training finished at $(date)"
echo "Pausing 10s before starting v3..."
sleep 10

echo ""
echo "============================================"
echo "  Starting v3 training"
echo "  Config: rank 16, all layers, cosine LR,"
echo "          adamw, grad_accum 4, 20K iters"
echo "============================================"
echo ""

source .venv/bin/activate
python3 finetune_tibeb.py --v3 2>&1 | tee "logs/v3_training_$(date '+%Y%m%d_%H%M%S').log"

echo ""
echo "v3 training finished at $(date)"
