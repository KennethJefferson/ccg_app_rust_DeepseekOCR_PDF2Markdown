import logging
import shutil
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File

from app import model
from app.pdf_pipeline import pdf_to_images
from app.schemas import ConvertResponse, HealthResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield


app = FastAPI(title="DeepSeek OCR Server", lifespan=lifespan)


@app.post("/convert", response_model=ConvertResponse)
async def convert(file: UploadFile = File(...)):
    temp_dir = None
    try:
        pdf_bytes = await file.read()
        temp_dir, image_paths = pdf_to_images(pdf_bytes)

        markdown_parts: list[str] = []
        for i, img_path in enumerate(image_paths):
            try:
                md = await model.infer(img_path)
                markdown_parts.append(md)
            except torch.cuda.OutOfMemoryError:
                logger.error("OOM on page %d, returning partial results", i)
                return ConvertResponse(
                    success=False,
                    markdown="\n\n---\n\n".join(markdown_parts),
                    pages_processed=len(markdown_parts),
                    error=f"GPU out of memory on page {i + 1}. {len(markdown_parts)}/{len(image_paths)} pages processed.",
                )

        return ConvertResponse(
            success=True,
            markdown="\n\n---\n\n".join(markdown_parts),
            pages_processed=len(markdown_parts),
        )
    except Exception as e:
        logger.exception("Conversion failed")
        return ConvertResponse(
            success=False,
            error=str(e),
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model.is_loaded() else "unavailable",
        model_loaded=model.is_loaded(),
        gpu_available=torch.cuda.is_available(),
    )
