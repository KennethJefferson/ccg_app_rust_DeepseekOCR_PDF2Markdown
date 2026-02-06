import asyncio
import functools
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2-3B"

_model: Optional[AutoModelForCausalLM] = None
_tokenizer: Optional[AutoTokenizer] = None
_semaphore = asyncio.Semaphore(1)


def is_loaded() -> bool:
    return _model is not None and _tokenizer is not None


def load_model() -> None:
    global _model, _tokenizer

    logger.info("Loading tokenizer from %s", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )

    logger.info("Loading model from %s", MODEL_NAME)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    logger.info("Model loaded successfully")


def _infer_sync(image_path: str) -> str:
    assert _model is not None and _tokenizer is not None, "Model not loaded"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "OCR the document to markdown."},
            ],
        }
    ]

    inputs = _tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, tokenize=True
    )

    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(_model.device)
        input_ids = inputs
    else:
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        input_ids = inputs.get("input_ids", inputs)

    with torch.no_grad():
        if isinstance(input_ids, dict):
            outputs = _model.generate(
                **input_ids,
                max_new_tokens=4096,
                do_sample=False,
            )
        else:
            outputs = _model.generate(
                input_ids,
                max_new_tokens=4096,
                do_sample=False,
            )

    if isinstance(inputs, torch.Tensor):
        prompt_len = inputs.shape[-1]
    else:
        prompt_len = inputs["input_ids"].shape[-1]

    generated_tokens = outputs[0][prompt_len:]
    result = _tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result.strip()


async def infer(image_path: str) -> str:
    async with _semaphore:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                None, functools.partial(_infer_sync, image_path)
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise
