import os
import tempfile

import fitz


def pdf_to_images(pdf_bytes: bytes, dpi: int = 68) -> tuple[str, list[str]]:
    """Convert PDF bytes to PNG images. Returns (temp_dir, list_of_image_paths)."""
    temp_dir = tempfile.mkdtemp(prefix="deepseek_ocr_")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    image_paths: list[str] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        path = os.path.join(temp_dir, f"page_{i:04d}.png")
        pix.save(path)
        image_paths.append(path)
    doc.close()
    return temp_dir, image_paths
