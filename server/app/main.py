import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File

from app import model
from app.schemas import ConvertResponse, HealthResponse

logger = logging.getLogger(__name__)

POOL_SIZE = int(os.environ.get("MARKER_POOL_SIZE", "4"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_models(POOL_SIZE)
    yield


app = FastAPI(title="DeepSeek OCR Server", lifespan=lifespan)


@app.post("/convert", response_model=ConvertResponse)
async def convert(file: UploadFile = File(...)):
    tmp_path = None
    try:
        pdf_bytes = await file.read()
        fd, tmp_path = tempfile.mkstemp(suffix=".pdf", prefix="marker_")
        os.write(fd, pdf_bytes)
        os.close(fd)

        start = time.monotonic()
        markdown, pages = await model.convert(tmp_path)
        elapsed = time.monotonic() - start

        if pages > 0:
            print(
                f"[TIMING] {pages} pages in {elapsed:.1f}s "
                f"(avg {elapsed / pages:.2f}s/page)",
                flush=True,
            )
        else:
            print(f"[TIMING] 0 pages in {elapsed:.1f}s", flush=True)

        return ConvertResponse(
            success=True,
            markdown=markdown,
            pages_processed=pages,
        )
    except Exception as e:
        logger.exception("Conversion failed")
        return ConvertResponse(success=False, error=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model.is_loaded() else "unavailable",
        model_loaded=model.is_loaded(),
        gpu_available=torch.cuda.is_available(),
        pool_size=model.pool_size(),
    )
