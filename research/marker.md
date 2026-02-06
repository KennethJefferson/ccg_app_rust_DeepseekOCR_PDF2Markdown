# Marker - PDF to Markdown Converter (Datalab / VikParuchuri)

**Repository:** https://github.com/datalab-to/marker (formerly VikParuchuri/marker)
**Latest Version:** 1.10.2 (January 31, 2026)
**PyPI:** https://pypi.org/project/marker-pdf/
**Python:** 3.10 - 3.14

---

## 1. Architecture & Pipeline

Marker is a pipeline of deep learning models (~800M parameters total, sub-billion) built on the
**Surya** model suite. The processing pipeline is:

1. **Text Extraction** - Extract embedded text from PDF (heuristics + pdftext)
2. **OCR** (if needed) - Detect whether OCR is required via heuristics, then run Surya OCR or
   Tesseract
3. **Layout Detection** - Detect page layout, columns, reading order (Surya layout model)
4. **Table Recognition** - Detect and structure tables (Surya table model)
5. **Block Cleaning & Formatting** - Clean/format each block using heuristics and Texify
   (for LaTeX equations)
6. **Post-processing** - Combine blocks, apply reading order, join spans, fix formatting

### Underlying Models

| Component | Architecture | Training |
|---|---|---|
| Text Detection | Modified **SegFormer** (reduced RAM) | 4x A6000, 3 days |
| Text Recognition (OCR) | Modified **Donut** model (GQA, MoE layer, UTF-16 decoding) | 4x A6000, 2 weeks |
| Layout Analysis | Surya layout model | Proprietary dataset |
| Table Recognition | Surya table model | FinTabNet + custom |
| Equation Conversion | **Texify** (LaTeX generation) | Proprietary |

Key design: Marker v2 moved to **block-level OCR** (vs. line-level in v1), which is slightly
slower but significantly more accurate.

### Optional LLM Enhancement

Pass `--use_llm` to enable an LLM alongside Marker for:
- Merging tables across pages
- Handling inline math
- Formatting complex tables
- Extracting values from forms

Default LLM: **Gemini 2.0 Flash**. Also supports Ollama for local/self-hosted models.

---

## 2. Conversion Speed & Benchmarks

### Official Benchmarks (from PyPI/README, H100 GPU)

| Tool | Avg Time Per Page | Heuristic Score | LLM Score |
|---|---|---|---|
| **Marker** | **2.84 sec** | **95.67** | - |
| Docling | 3.70 sec | 86.71 | - |
| Mathpix (cloud) | 6.36 sec | 86.43 | - |
| LlamaParse (cloud) | 23.35 sec | 84.24 | - |

### Throughput Numbers

| Configuration | Throughput |
|---|---|
| H100, single process | ~25 pages/sec |
| H100, 22 parallel processes | ~122 pages/sec (0.18 sec/page effective) |
| A6000, ~24 parallel docs | ~24 docs simultaneously |
| CPU only (no GPU) | Very slow - minutes per page |
| CPU, no OCR | ~60% faster than CPU with OCR |

### Real-World Reports

- **H100, 10k pages, force_ocr**: 5 hours 31 minutes = ~30 pages/min = ~0.5 pages/sec
  (force_ocr is significantly slower than default)
- **CPU-only (HuggingFace free tier, 2 vCPU/16GB RAM)**: 42-page PDF took ~5 hours
- **CPU-only, 9-page PDF**: Reported ~10 minutes total
- **Marker is 4x faster than Nougat** and more accurate outside arXiv

### Table Extraction Accuracy (FinTabNet)

| Mode | Score |
|---|---|
| Marker (default) | 0.816 |
| Marker + `--use_llm` | **0.907** |
| Gemini alone | 0.829 |

### Datalab Hosted API Speed

- ~15 seconds for a 250-page PDF
- 3-4 pages/second throughput
- Three modes: "fast" (low latency), "balanced" (default), "accurate" (complex layouts)

---

## 3. GPU / VRAM Requirements

### VRAM Usage

| Metric | VRAM |
|---|---|
| Default batch sizes (single worker) | ~3 GB |
| Average per worker (multi-worker) | ~3.5 GB |
| Peak per worker | ~5 GB |
| Surya OCR recognition (batch 256) | ~12.8 GB |
| Surya layout analysis (batch 32) | ~7 GB |

### Scaling Formula

Parallelism is capped at: `INFERENCE_RAM / VRAM_PER_TASK`

- A6000 (48 GB): ~24 parallel documents
- H100 (80 GB): ~22 parallel processes

### Supported Devices

- **CUDA** (NVIDIA GPUs) - recommended
- **MPS** (Apple Metal) - supported but slower
- **CPU** - supported but very slow, not recommended for production

Override with: `TORCH_DEVICE=cuda` (or `cpu`, `mps`)

---

## 5. RTX 3090 (24GB VRAM) - Can It Run?

**Yes, absolutely.** The RTX 3090 with 24GB VRAM is well-suited for Marker.

### Estimated Configuration on RTX 3090

| Setting | Estimate |
|---|---|
| Single worker VRAM | 3-5 GB |
| Max parallel workers | ~4-6 workers (24GB / 5GB peak) |
| Expected throughput | ~5-12 pages/sec (estimated) |

### Reasoning

- Marker's default batch sizes only need ~3 GB VRAM
- With 24 GB, you have headroom for 4-6 parallel workers
- The RTX 3090 has roughly 1/5th to 1/10th the inference throughput of an H100
- H100 does 122 pages/sec with 22 workers; RTX 3090 with ~5 workers should do roughly
  5-15 pages/sec depending on document complexity
- **No published RTX 3090 benchmarks exist** - you would need to run `benchmark.py` yourself

### Caveats

- If using `--use_llm` with a local model, add that model's VRAM to the budget
- `--force_ocr` increases VRAM usage and slows processing significantly
- Reduce `--workers` if you get OOM errors
- Surya OCR at default batch size (256) needs ~12.8 GB alone, but batch size is configurable

---

## 6. Multi-Page PDF Handling

- Marker processes multi-page PDFs natively as a single unit
- Page selection via `--page_range "0,5-10,20"` (comma-separated pages and ranges)
- Layout detection and reading order are computed per page
- With `--use_llm`, tables can be **merged across page boundaries**
- For very long PDFs causing memory issues, the recommendation is to split into smaller files
- Images are extracted and linked in the output automatically

---

## 7. Batch Processing

### Single Machine

```bash
# Convert all PDFs in a folder
marker /path/to/input/folder --output_dir /path/to/output

# Control parallelism
marker /path/to/input/folder --workers 4
```

### Multi-GPU

```bash
NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out
```

- `NUM_DEVICES`: Number of GPUs to use
- `NUM_WORKERS`: Total parallel processes across all GPUs
- `--workers`: Number of PDFs to convert simultaneously (default: 1)
- Parallelism is auto-capped by available VRAM

### Programmatic / API

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

models = create_model_dict()
converter = PdfConverter(artifact_dict=models)
rendered = converter("path/to/file.pdf")
```

### Datalab Hosted API

- REST API at datalab.to
- Supports PDF, images, PPTX, DOCX, XLSX, HTML, EPUB
- Results available for 1 hour after processing

---

## 8. Known Limitations

### Accuracy Limitations

- **Equations**: Will not convert 100% of equations to LaTeX. Must detect then convert,
  which is less complete than Nougat (but Nougat hallucinates more on non-arXiv content)
- **Tables**: Not always 100% correct - text can end up in wrong columns, whitespace/indentation
  not always preserved, not all lines/spans joined properly
- **Complex layouts**: Nested tables and forms may not work well
- **Headings**: PDF document outline / heading detection not always accurate, especially
  multi-level headings and section ordering
- **Inline math**: Requires `--force_ocr` flag for best results

### Performance Limitations

- **CPU-only is very slow** - models are ~800M parameters, need GPU for practical use
- **force_ocr is significantly slower** than default mode
- **Text recognition is the bottleneck** - layout detection is fast, OCR is slow
- **Memory**: Very long PDFs may cause OOM; split into smaller files as workaround

### Format Limitations

- Image quality in output depends on PDF source quality
- Some PDF types with garbled text need `--force_ocr` or `--strip_existing_ocr`
- Headers/footers may be ignored or misclassified in some documents

### Workarounds

- `--force_ocr` or `OCR_ALL_PAGES=true` for garbled text or bad table layouts
- `--strip_existing_ocr` for PDFs with bad embedded OCR layers
- `--use_llm` significantly improves table and form accuracy
- `--disable_image_extraction` to skip images and speed up processing

---

## 9. License & Pricing

### Open Source License

| Component | License |
|---|---|
| Code | **GPL-3.0** |
| Model weights | Modified **AI Pubs Open Rail-M** |

### Model Weight License Terms

- **Free** for: research, personal use, startups under $2M funding/revenue
- **Commercial license required** for: companies above $2M threshold
- Contact Datalab for commercial licensing to remove GPL requirements

### Datalab Hosted API Pricing

- Free signup with included credits to test
- Structured extraction with page_schema: ~$6 per 1,000 pages
- Standard conversion: ~1.5 cents per page (based on API response examples)
- 1/4th the price of leading cloud competitors (per Datalab's claims)
- On-prem licensing available for enterprise
- Custom enterprise quotes available

### Comparison: Self-Hosted vs API

| Factor | Self-Hosted | Datalab API |
|---|---|---|
| Cost | GPU hardware + electricity | ~$0.006-0.015/page |
| Speed | Depends on hardware | ~15 sec for 250 pages |
| Setup | Complex (Python, CUDA, models) | REST API call |
| License | GPL + model weight restrictions | Included |
| Control | Full | Limited to API options |

---

## 10. Installation Complexity

### Quick Install

```bash
pip install marker-pdf          # PDF only
pip install marker-pdf[full]    # All document types (DOCX, PPTX, etc.)
```

### Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU support)
- ~3-5 GB disk space for model weights (downloaded on first run)

### Key Dependencies

surya-ocr, torch, transformers, pdftext, pillow, pydantic, markdownify, google-genai
(for LLM mode), anthropic (for LLM mode), openai (for LLM mode)

### Installation Gotchas

- May need to install CPU-specific PyTorch first if no GPU: `pip install torch --index-url ...`
- Python 3.13+ had reported issues in some environments (likely resolved in 1.10.x)
- Model weights auto-download on first run (~3-5 GB from HuggingFace)
- For LLM features, need API keys (GOOGLE_API_KEY for Gemini, or Ollama running locally)

### Docker

Community Docker images available: `savatar101/marker-api`

### Complexity Rating: **Moderate**

Simple `pip install` works for basic usage. Complexity increases with:
- CUDA/GPU setup (driver compatibility)
- Multi-GPU configuration
- LLM integration (API keys, local model hosting)
- Production deployment (worker tuning, memory management)

---

## Summary Assessment

### Strengths

- Best-in-class open-source accuracy (heuristic score 95.67 vs next-best 86.71)
- Fast on GPU hardware (2.84 sec/page single process, 122 pages/sec at scale)
- Broad format support (PDF, images, DOCX, PPTX, XLSX, HTML, EPUB)
- 90+ language OCR support via Surya
- LLM enhancement option for complex documents
- Active development (v1.10.2, Jan 2026)
- Multiple output formats (markdown, HTML, JSON, chunks for RAG)

### Weaknesses

- GPL license limits commercial use without paid license
- CPU-only performance is impractical (minutes per page)
- Table accuracy still imperfect without LLM mode
- No published benchmarks for consumer GPUs (RTX 3090, 4090, etc.)
- Equation detection is lossy compared to specialized tools
- Complex nested layouts remain challenging

### For RTX 3090 Specifically

The RTX 3090 is a strong choice for self-hosted Marker. With 24GB VRAM, you can run
multiple parallel workers and likely achieve 5-15 pages/second throughput. This is
a practical setup for small-to-medium batch processing (hundreds to low thousands of pages).
For higher volumes, consider the Datalab API or multi-GPU setups.

---

## Sources

- [Marker GitHub Repository](https://github.com/datalab-to/marker)
- [marker-pdf on PyPI](https://pypi.org/project/marker-pdf/)
- [Surya OCR GitHub](https://github.com/datalab-to/surya)
- [Datalab Platform](https://www.datalab.to/platform)
- [Datalab Pricing](https://www.datalab.to/pricing)
- [Datalab Documentation - Marker API](https://documentation.datalab.to/docs/recipes/marker/conversion-api-overview)
- [GitHub Issue #943 - Conversion Speed](https://github.com/datalab-to/marker/issues/943)
- [Modal + Datalab Deployment](https://modal.com/blog/datalab-and-modal)
- [Marker DeepWiki - Model Management](https://deepwiki.com/datalab-to/marker/5.7-model-management)
- [HackerNews Benchmark Discussion](https://news.ycombinator.com/item?id=43285912)
- [PDF Conversion Quality Showdown](https://www.snaps2pdf.com/2025/10/pdf-conversion-quality-showdown.html)
