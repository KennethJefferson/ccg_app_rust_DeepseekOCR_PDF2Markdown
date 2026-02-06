#!/bin/bash
set -e

VENV=/workspace/venv
SERVER=/workspace/deepseek-ocr-server
MODEL=/workspace/models/DeepSeek-OCR-2

# Create venv + install deps if not already present
if [ ! -f "$VENV/bin/python" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV"
    echo "Installing dependencies..."
    "$VENV/bin/pip" install -r "$SERVER/requirements.txt"
    "$VENV/bin/pip" install torchvision==0.21.0 --no-deps
fi

# Download model if not already present
if [ ! -f "$MODEL/config.json" ]; then
    echo "Downloading model..."
    "$VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='deepseek-ai/DeepSeek-OCR-2', local_dir='$MODEL')
"
fi

echo "Starting server (model: $MODEL)..."
cd "$SERVER"
"$VENV/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --workers 1
