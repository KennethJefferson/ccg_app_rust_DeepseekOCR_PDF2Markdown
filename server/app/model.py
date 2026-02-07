import asyncio
import functools
import logging
from typing import Optional

import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

logger = logging.getLogger(__name__)

_converter: Optional[PdfConverter] = None


def is_loaded() -> bool:
    return _converter is not None


def load_model() -> None:
    global _converter
    logger.info("Loading Marker models...")
    models = create_model_dict()
    _converter = PdfConverter(artifact_dict=models)
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info(f"Marker models loaded. VRAM used: {vram_mb:.0f} MB")


def _convert_sync(pdf_path: str) -> tuple[str, int]:
    assert _converter is not None, "Model not loaded"
    rendered = _converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    page_count = len(rendered.metadata.get("page_stats", []))

    # Free CUDA cache after each conversion to prevent VRAM fragmentation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return text, page_count


async def convert(pdf_path: str) -> tuple[str, int]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, functools.partial(_convert_sync, pdf_path)
    )
