#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Pre-downloading model..."
python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True); AutoModel.from_pretrained('deepseek-ai/DeepSeek-OCR-2', trust_remote_code=True, use_safetensors=True)"

echo "Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
