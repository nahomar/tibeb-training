#!/bin/bash
# Run synthetic data generation with proper logging.
# Usage: ./scripts/gen_run.sh [category]  (default: financial)
# Categories: financial, general, voice, safety, deep, all

cd "$(dirname "$0")/.."
source .venv/bin/activate
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY before running}"
export PYTHONUNBUFFERED=1

CAT="${1:-financial}"
LOG="logs/gen_${CAT}_$(date '+%Y%m%d_%H%M%S').log"
mkdir -p logs

echo "Starting $CAT generation, logging to $LOG"
python3 -u scripts/gen_simple.py "$CAT" 2>&1 | tee "$LOG"
echo "Done at $(date)"
