#!/bin/bash
set -e

VENV=/workspace/venv
SERVER=/workspace/pdf2md-server
export HF_HOME=/workspace/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Cleanup stale processes from previous runs ---
echo "Cleaning up stale processes..."
# Kill any existing uvicorn/python workers on port 8000
for pid in $(pgrep -f 'uvicorn app.main' 2>/dev/null || true); do
    kill -9 "$pid" 2>/dev/null || true
done
# Kill orphaned python3 processes left by crashed uvicorn workers
for pid in $(pgrep python3 2>/dev/null || true); do
    kill -9 "$pid" 2>/dev/null || true
done
sleep 2
# Verify port is free
if ss -tlnp 2>/dev/null | grep -q ':8000 '; then
    echo "ERROR: Port 8000 still in use after cleanup"
    ss -tlnp | grep ':8000 '
    exit 1
fi
echo "Cleanup complete. Port 8000 free."

# Clear __pycache__ to prevent stale bytecode
rm -rf "$SERVER/app/__pycache__"

# Clean orphaned temp files from previous crashed workers
rm -f /tmp/marker_*.pdf

# Ensure datalab/surya model cache symlinks to persistent storage
if [ -d /workspace/datalab_cache ] && [ ! -L /root/.cache/datalab ]; then
    rm -rf /root/.cache/datalab
    mkdir -p /root/.cache
    ln -s /workspace/datalab_cache /root/.cache/datalab
fi

# Recreate venv if marker-pdf not installed
if ! "$VENV/bin/pip" show marker-pdf &>/dev/null 2>&1; then
    echo "Creating virtual environment..."
    rm -rf "$VENV"
    python3 -m venv "$VENV"
    echo "Installing dependencies..."
    "$VENV/bin/pip" install -r "$SERVER/requirements.txt"
fi

WORKERS=${MARKER_WORKERS:-4}
echo "Starting server with $WORKERS workers..."
cd "$SERVER"
"$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --workers "$WORKERS"
