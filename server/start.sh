#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Pre-downloading model..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-OCR-2-3B', trust_remote_code=True); AutoModelForCausalLM.from_pretrained('deepseek-ai/DeepSeek-OCR-2-3B', trust_remote_code=True, torch_dtype='auto')"

echo "Starting server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
