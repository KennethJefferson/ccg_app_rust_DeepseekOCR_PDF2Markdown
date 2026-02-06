import asyncio
import functools
import logging
from typing import Optional

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

_model: Optional[AutoModel] = None
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
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        _attn_implementation="flash_attention_2",
        use_safetensors=True,
    )
    _model = model.eval().cuda().to(torch.bfloat16)
    logger.info("Model loaded successfully")


def _infer_sync(image_path: str) -> str:
    assert _model is not None and _tokenizer is not None, "Model not loaded"

    with torch.inference_mode():
        result = _model.infer(
            _tokenizer,
            prompt=PROMPT,
            image_file=image_path,
            output_path="",
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=False,
            eval_mode=True,
        )

    return result if isinstance(result, str) else str(result)


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
