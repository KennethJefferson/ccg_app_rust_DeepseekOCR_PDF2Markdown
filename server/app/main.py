import glob
import hashlib
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, Header, UploadFile, File

from app import model
from app.schemas import ConvertResponse, HealthResponse

logger = logging.getLogger(__name__)

FATAL_CUDA_KEYWORDS = ["device-side assert", "CUDA error", "CUDA out of memory"]


def _is_fatal_cuda_error(error: Exception) -> bool:
    msg = str(error)
    return any(kw in msg for kw in FATAL_CUDA_KEYWORDS)


def _cleanup_stale_temp_files() -> None:
    """Remove orphaned marker temp files from previous crashed workers."""
    for f in glob.glob("/tmp/marker_*.pdf"):
        try:
            os.unlink(f)
        except OSError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _cleanup_stale_temp_files()
    model.load_model()
    yield


app = FastAPI(title="PDF2Markdown Server", lifespan=lifespan)


@app.post("/convert", response_model=ConvertResponse)
async def convert(
    file: UploadFile = File(...),
    x_file_md5: str | None = Header(None),
):
    tmp_path = None
    try:
        pdf_bytes = await file.read()
        received_size = len(pdf_bytes)
        print(f"[UPLOAD] {file.filename}: {received_size} bytes", flush=True)

        # Verify upload integrity if client sent MD5 hash
        if x_file_md5:
            actual_md5 = hashlib.md5(pdf_bytes).hexdigest()
            if actual_md5 != x_file_md5:
                return ConvertResponse(
                    success=False,
                    error=f"Upload corrupted: expected MD5 {x_file_md5}, got {actual_md5} ({received_size} bytes)",
                )

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

        # Clean up CUDA cache even on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fatal CUDA errors corrupt the worker's GPU context permanently.
        # Kill the worker so uvicorn restarts it with a fresh CUDA context.
        if _is_fatal_cuda_error(e):
            logger.critical(f"Fatal CUDA error, worker exiting for restart: {e}")
            # Clean up temp file before exit
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os._exit(1)

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
    )
