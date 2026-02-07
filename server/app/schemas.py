from pydantic import BaseModel
from typing import Optional


class ConvertResponse(BaseModel):
    success: bool
    markdown: str = ""
    pages_processed: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    cuda_healthy: bool = False
