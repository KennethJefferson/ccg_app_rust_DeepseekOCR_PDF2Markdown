import asyncio
import functools
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

logger = logging.getLogger(__name__)

_pool: Optional[asyncio.Queue] = None
_pool_size: int = 0
_executor: Optional[ThreadPoolExecutor] = None


def is_loaded() -> bool:
    return _pool is not None and _pool_size > 0


def pool_size() -> int:
    return _pool_size


def load_models(n: int = 4) -> None:
    global _pool, _pool_size, _executor
    logger.info(f"Loading {n} Marker model instance(s)...")

    models = create_model_dict()
    _pool = asyncio.Queue(maxsize=n)
    _executor = ThreadPoolExecutor(max_workers=n)

    for i in range(n):
        converter = PdfConverter(artifact_dict=models)
        _pool.put_nowait(converter)
        logger.info(f"  Instance {i + 1}/{n} ready")

    _pool_size = n
    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
    logger.info(f"All {n} instances loaded. VRAM used: {vram_mb:.0f} MB")


def _convert_sync(converter: PdfConverter, pdf_path: str) -> tuple[str, int]:
    rendered = converter(pdf_path)
    text, _, _ = text_from_rendered(rendered)
    page_count = len(rendered.metadata.get("page_stats", []))
    return text, page_count


async def convert(pdf_path: str) -> tuple[str, int]:
    assert _pool is not None, "Model pool not initialized"
    assert _executor is not None, "Executor not initialized"

    converter = await _pool.get()
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor, functools.partial(_convert_sync, converter, pdf_path)
        )
        return result
    finally:
        await _pool.put(converter)
