#!/bin/bash
set -e

VENV=/workspace/venv
SERVER=/workspace/pdf2md-server
export HF_HOME=/workspace/hf_cache

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

echo "Starting server..."
cd "$SERVER"
"$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --workers 1
