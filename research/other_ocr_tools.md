# OCR Tools for PDF-to-Markdown: Comprehensive Research & Comparison

> Research date: February 2026. Benchmarks collected from official repos, papers, and third-party evaluations.

---

## Quick Comparison Table

| Tool | Architecture | Params | Speed (sec/page) | GPU VRAM | PDF-to-MD Quality | License | Active? |
|------|-------------|--------|-------------------|----------|-------------------|---------|---------|
| **Surya / Marker** | Foundation models + pipeline | Multiple models | ~0.05-0.4 (A10 GPU) | 5-20 GB | Excellent -- structured MD with tables, formulas | GPL-3.0 code / restricted weights | Yes (19.2k stars) |
| **Nougat** | Encoder-decoder VIT (Donut-based) | 250M / 350M | ~5-20 (single page, GPU) | ~6-8 GB (base) | Good for academic papers; LaTeX math | MIT code / CC-BY-NC weights | Stale (last commit Aug 2023) |
| **GOT-OCR 2.0** | Vision encoder + LLM decoder (Qwen) | 580M | ~2-5 (estimated, GPU) | ~4-8 GB (FP16) | Good for mixed content; MD + LaTeX output | Apache 2.0 code / CC-BY-NC data | Moderate (last ~Sep 2024) |
| **Mistral OCR 3** | Proprietary vision LLM | Unknown (API) | ~0.03 (2000 pages/min) | N/A (cloud API) | Excellent -- 96.6% table accuracy, strong handwriting | Proprietary (API only) | Yes (Dec 2025 release) |
| **Tesseract 5** | LSTM + legacy OCR engine | N/A | ~0.9-2.0 (CPU only) | 0 (CPU only) | Poor for markdown; plain text only, no layout | Apache 2.0 | Yes (maintained) |

---

## 1. Surya OCR / Marker

**Repository:** [github.com/datalab-to/surya](https://github.com/datalab-to/surya) (19.2k stars, 1.3k forks)
**Marker:** [github.com/datalab-to/marker](https://github.com/datalab-to/marker)

### Architecture / Approach

Surya is a modular OCR toolkit that provides multiple foundation models for different subtasks:

- **Text detection** -- identifies text line regions
- **Layout analysis** -- classifies page regions (text, table, figure, heading, etc.)
- **Reading order** -- determines correct reading sequence
- **Table recognition** -- structured table extraction
- **OCR recognition** -- character-level text extraction in 90+ languages

Marker is the higher-level pipeline that combines Surya's models to produce structured output (Markdown, JSON, HTML). It optionally integrates LLMs for complex tables and formulas.

### Speed Benchmarks (A10 GPU)

| Subtask | Seconds/Page | Notes |
|---------|-------------|-------|
| Detection | 0.108 | Batch size 36 |
| Layout | 0.273 | Batch size 32 |
| Table recognition | 0.022 | Batch size 64 |
| OCR recognition | varies | Batch size 512, depends on text density |
| **Marker end-to-end** | **~0.05-0.4** | 20-120 pages/sec on H100 (batch mode) |

Marker's throughput is highly variable depending on document complexity. On an H100, batch processing achieves 20-120 pages/second. A benchmark of 10,000 pages on a single H100 took 5 hours 31 minutes (~0.5 pages/sec sustained with overhead, or ~$14.84 total cost).

The OCR model supports `torch.compile()` for an approximately 15% speedup.

### GPU Requirements

| Component | VRAM (default batch size) |
|-----------|--------------------------|
| Detection | ~16 GB (batch 36) |
| Layout | ~7 GB (batch 32) |
| Table recognition | ~10 GB (batch 64) |
| OCR recognition | ~20 GB (batch 512) |
| **Marker (full pipeline)** | **~5 GB peak, 3.5 GB average per worker** |

Each batch item uses approximately 220-280 MB of VRAM. Batch sizes are tunable -- you can reduce VRAM by lowering batch size at the cost of speed.

Benchmarks use A6000 (48 GB) for Surya; Marker works on consumer GPUs with reduced batch sizes.

### Accuracy

- Benchmarks favorably against cloud OCR services (Google, AWS, Azure)
- Marker scored 4.41 in LLM-as-judge evaluation (375 samples), comparable to Mistral OCR at 4.32
- Excellent structural preservation: headings, lists, tables, footnotes, reading order
- 90+ language support

### License

- **Code:** GPL-3.0
- **Model weights:** Modified AI Pubs Open Rail-M -- free for research, personal use, and startups under $2M funding/revenue. Larger commercial entities need a paid license.
- **Commercial use:** Restricted for companies above $2M funding. On-prem commercial licensing available via [datalab.to](https://www.datalab.to/platform).

### Maintenance

Actively maintained. 952 commits, frequent updates through 2025-2026. Created by Vik Paruchuri (formerly VikParuchuri/surya, now under datalab-to org).

### Key Takeaway

Marker/Surya is the best open-source option for general PDF-to-Markdown conversion. Fast on GPU, excellent structural output, but the GPL + weight license restrictions may be a concern for commercial use.

---

## 2. Nougat (Meta / Facebook Research)

**Repository:** [github.com/facebookresearch/nougat](https://github.com/facebookresearch/nougat) (9.8k stars)
**Paper:** [arxiv.org/abs/2308.13418](https://arxiv.org/abs/2308.13418) (August 2023)

### Architecture / Approach

Nougat (Neural Optical Understanding for Academic Documents) is an end-to-end encoder-decoder transformer model built on the [Donut](https://github.com/clovaai/donut) architecture. It converts rasterized document page images directly to Mathpix Markdown format -- no traditional OCR step, no reliance on embedded text.

- **Encoder:** Swin Transformer for visual feature extraction
- **Decoder:** BART-based autoregressive text decoder (10 layers)
- **Two variants:** `0.1.0-small` (250M params) and `0.1.0-base` (350M params)
- **Output format:** Mathpix Markdown (with LaTeX math, tables)

The model processes one page at a time as an image and autoregressively generates the full markup output.

### Speed Benchmarks

Nougat is notably slow compared to pipeline-based approaches:

| Metric | Value | Notes |
|--------|-------|-------|
| Mean generation time (base) | ~19.5 seconds per page | Without inference optimization, A100 |
| Token generation | ~1400 tokens per page | Greedy decoding |
| Comparison to GROBID | ~200x slower | GROBID processes at 10.6 PDF/sec |

Speed varies significantly with page text density. Math-heavy pages take longer due to LaTeX token generation. With optimizations (half precision, batch decoding), speed improves but remains in the 5-10 sec/page range.

### GPU Requirements

- **Base model (350M):** ~6-8 GB VRAM for inference (FP16)
- **Small model (250M):** ~4-6 GB VRAM
- CPU inference is technically possible but impractical (minutes per page)
- Known issues with false positives in failure detection on CPU or older GPUs

### Accuracy

- Designed specifically for academic/scientific documents (arXiv, PMC papers)
- Excellent at LaTeX math expression preservation
- Handles tables in academic papers well
- **Weakness:** Struggles with non-academic documents -- business docs, invoices, multi-column layouts, handwriting
- Can hallucinate on pages with heavy figures or unusual layouts (repetitive token generation)
- The paper reports competitive edit distance scores against GROBID on their test set

### License

- **Code:** MIT License
- **Model weights:** CC-BY-NC (Creative Commons Non-Commercial)
- **Commercial use:** Weights are non-commercial. Code is permissive. You cannot use the pretrained weights commercially without separate licensing from Meta.

### Maintenance

**Effectively unmaintained.** Last significant activity was August 2023. 78 commits, 14 contributors, 2 releases. The repository has not received meaningful updates in over 2 years. Issues pile up without responses.

### Key Takeaway

Nougat is a pioneering model for academic PDF-to-LaTeX/Markdown conversion but is slow, narrow in scope (academic papers only), and abandoned. Not recommended for new projects unless you specifically need LaTeX math extraction from arXiv-style papers and are willing to deal with the speed penalty.

---

## 3. GOT-OCR 2.0 (General OCR Theory)

**Repository:** [github.com/Ucas-HaoranWei/GOT-OCR2.0](https://github.com/Ucas-HaoranWei/GOT-OCR2.0) (8.1k stars, 701 forks)
**Paper:** [arxiv.org/abs/2409.01704](https://arxiv.org/abs/2409.01704) (September 2024)

### Architecture / Approach

GOT-OCR 2.0 introduces a "unified end-to-end" approach to OCR using a vision-language model architecture:

- **Vision encoder:** High-compression image encoder
- **Projector:** Bridges vision and language components
- **LLM decoder:** Based on Qwen language model
- **Total parameters:** 580M
- **Built on:** Vary architecture codebase

The model handles diverse OCR tasks via prompt-based instructions: plain text extraction, formatted text (Markdown/LaTeX), fine-grained region OCR, and even specialized symbols (sheet music, molecular formulas, geometric shapes).

Supports both single-page and multi-page (tiled) input with batched inference via HuggingFace Transformers integration.

### Speed Benchmarks

Specific published benchmarks are limited, but available data suggests:

| Metric | Value | Notes |
|--------|-------|-------|
| Estimated inference | ~2-5 sec/page | GPU required for real-time |
| Token usage | ~256+ vision tokens per image | Higher than some competing models |
| vs. PaddleOCR | Slower | Higher latency than modular pipelines |

GOT-OCR 2.0 requires GPU for practical use. Inference latency is higher than pipeline-based systems (PaddleOCR, Surya/Marker) but lower than Nougat.

Flash-Attention is recommended for optimization.

### GPU Requirements

- **Estimated VRAM:** ~4-8 GB for FP16 inference (580M params)
- **Environment:** CUDA 11.8+, PyTorch 2.0.1+
- **Optimization:** Flash-Attention support for reduced memory and faster inference
- The relatively small parameter count (580M) makes it feasible on consumer GPUs (RTX 3060+)

### Accuracy

- Scored **61.00** on specialized OCR benchmarks (top among specialized models in its class as of late 2024)
- Handles: plain text, tables, charts, equations, formatted output
- Supports Markdown and LaTeX output formats via prompting
- Competitive with much larger models on document OCR tasks
- **Weakness:** Not specifically optimized for full-page PDF layout preservation; better at region-level OCR

### License

- **Code:** Apache 2.0
- **Data:** CC-BY-NC 4.0
- **Practical impact:** Code is permissively licensed, but the training data license restricts commercial use of models trained on their data.

### Maintenance

Moderate activity. 101 commits, last significant update around September 2024. The model is integrated into HuggingFace Transformers, which provides ongoing compatibility. The original repo is not as actively maintained as Surya/Marker.

### Key Takeaway

GOT-OCR 2.0 is an interesting middle ground -- small enough for consumer GPUs, versatile prompt-based OCR, and Apache-licensed code. However, it lacks the full PDF-to-Markdown pipeline that Marker provides. You would need to build the page-level orchestration yourself. Best suited for document understanding tasks rather than bulk PDF conversion.

---

## 4. Mistral OCR / Pixtral (Vision LLMs for OCR)

**Mistral OCR 3:** [mistral.ai/news/mistral-ocr-3](https://mistral.ai/news/mistral-ocr-3) (December 2025)
**Pixtral 12B/Large:** [mistral.ai/news/pixtral-12b](https://mistral.ai/news/pixtral-12b)

### Architecture / Approach

Mistral offers two relevant products:

**Pixtral (general multimodal LLM):**
- Pixtral 12B: 12B parameter multimodal model (first Mistral vision model)
- Pixtral Large: 124B parameters (123B text decoder + 1B vision encoder)
- General-purpose vision-language model, not OCR-specific

**Mistral OCR 3 (specialized document AI):**
- Purpose-built for document parsing, table reconstruction, and markdown output
- Smaller and faster than Pixtral, optimized for document tasks
- Handles: forms, scanned documents, tables, handwriting, multi-page PDFs
- Outputs structured Markdown with layout preservation

### Speed Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | ~2,000 pages/minute | Single node, Mistral OCR 3 |
| Per page | ~0.03 seconds | At peak throughput |
| Batch API | Same speed, 50% cost discount | Async processing |

This is by far the fastest option in this comparison, but it requires API access.

### GPU Requirements

- **N/A for end users** -- cloud API only
- No self-hosted option available
- Pixtral 12B can be self-hosted (12B params, ~24 GB VRAM for FP16), but it is a general VLM, not the specialized OCR model

### Accuracy

| Task | Mistral OCR 3 | Competitor |
|------|---------------|------------|
| Handwriting | 88.9% | Azure 78.2%, DeepSeek 57.2% |
| Tables | 96.6% | AWS Textract 84.8% |
| Overall win rate | 74% | vs. Mistral OCR 2 (internal benchmark) |

Strong performance on structured documents, forms, and handwriting. The 74% win rate claim is against their own previous version, not independent benchmarks -- take with appropriate skepticism.

### Pricing

| Tier | Cost |
|------|------|
| Standard API | $2 per 1,000 pages |
| Batch API | $1 per 1,000 pages |
| **Comparison:** AWS Textract | $65 per 1,000 pages (forms+tables) |
| **Comparison:** Google Doc AI | $30-45 per 1,000 pages |
| **Comparison:** Azure Form Recognizer | $1.50 per 1,000 pages (basic) |

### License

- **Proprietary** -- API access only via Mistral platform
- No self-hosted deployment for OCR 3
- Pixtral 12B weights are available under Apache 2.0, but that is the general VLM, not the OCR-specific model

### Maintenance

Actively developed. OCR 3 released December 2025 with significant improvements over OCR 2 (March 2025). Mistral is well-funded and actively iterating.

### Key Takeaway

If you can accept API dependency and per-page costs, Mistral OCR 3 offers the best speed and competitive accuracy. At $1-2/1000 pages, it is very cost-effective for batch processing. However, no self-hosting means no offline use, no data privacy guarantees beyond Mistral's policies, and ongoing cost scaling with volume.

---

## 5. Tesseract 5 (+ Layout Analysis)

**Repository:** [github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
**Version:** 5.x (stable since November 2021)

### Architecture / Approach

Tesseract is the classic open-source OCR engine, now in its 5th major version:

- **Recognition engine:** LSTM-based neural network (replaced legacy engine in v4+)
- **Layout analysis:** Built-in page segmentation (columns, blocks, paragraphs)
- **Output formats:** Plain text, hOCR (HTML with bounding boxes), ALTO XML, TSV
- **No native Markdown output** -- requires post-processing pipeline

For PDF-to-Markdown, Tesseract must be combined with external tools:
- Layout analysis: LayoutParser, Detectron2, or similar
- Table extraction: Separate table detection model
- Markdown generation: Custom post-processing code
- This means significant engineering effort to build a competitive pipeline

### Speed Benchmarks

| Metric | Value | Notes |
|--------|-------|-------|
| Single page (CPU) | ~0.9-2.0 seconds | A4 scan, 12-core CPU |
| Batch (200 pages, CPU) | ~51 pages/minute | Mixed document types |
| Throughput estimate | ~0.7-1.1 pages/second | CPU only, single thread |

Tesseract is CPU-only. No GPU acceleration. Speed scales linearly with CPU cores for parallel processing, but individual page latency remains high.

### GPU Requirements

- **None** -- CPU only
- No CUDA/GPU support
- Scales horizontally across CPU cores
- Low memory footprint (~100-500 MB RAM)

### Accuracy

- **Clean printed text:** ~95%+ character accuracy on high-quality scans
- **Complex layouts:** Poor -- multi-column, tables, mixed content cause significant errors
- **Handwriting:** Very poor
- **Math/formulas:** Not supported
- **Tables:** No native table structure recognition
- **vs. modern VLMs:** Significantly outperformed by all models in this comparison on complex documents

Tesseract works well as a baseline for simple, clean, single-column printed documents. It falls apart on anything more complex.

### License

- **Apache 2.0** -- fully permissive, no restrictions on commercial use
- This is the most permissive license in the comparison

### Maintenance

Actively maintained by Stefan Weil and Zdenko Podobny. Regular bugfix releases on the 5.x branch. However, architectural innovation has stalled -- no major new features since v4's LSTM engine. The project receives maintenance updates, not research advances.

### Key Takeaway

Tesseract is the baseline. Free, permissive, runs anywhere (no GPU needed), but produces plain text only. For PDF-to-Markdown, you need to build a significant post-processing pipeline, and the result will still be worse than any VLM-based approach on complex documents. Use it only when: (1) you need Apache 2.0 licensing, (2) you have simple documents, or (3) you have no GPU access and need offline processing.

---

## 6. Notable Mentions (Late 2025 - Early 2026)

The OCR landscape evolved rapidly in late 2025. Several newer models are worth tracking:

| Model | Params | Pages/sec (H100) | olmOCR-Bench Score | Notes |
|-------|--------|-------------------|-------------------|-------|
| **Chandra** (Datalab) | 9B | 1.29 | 83.1 | Highest accuracy; from Surya/Marker creator |
| **OlmOCR-2** (Allen AI) | ~7.7B | 1.78 | 82.4 | Strong accuracy, open weights |
| **PaddleOCR-VL** | 0.9B | 2.20 | 80.0 | Very small, efficient |
| **DeepSeek-OCR** | 3B (570M active, MoE) | 4.65 | 75.7 | Fast via MoE architecture |
| **LightOn OCR** | 1B | 5.55 | 76.1 | Fastest open model |
| **OCRFlux-3B** | 3B | -- | SOTA on benchmarks | Runs on GTX 3090, cross-page merging |
| **Nanonets OCR 2** | 4B | 1.12 | 69.5 | Fine-tuned Qwen2.5-VL |
| **Dolphin** (ByteDance) | -- | -- | -- | VIT OCR + layout, MD/JSON output |
| **MinerU** (OpenDataLab) | -- | -- | -- | Multi-model pipeline, 84 languages |

These VLM-based models represent the current state of the art. October 2025 alone saw six major open-source OCR model releases.

---

## Decision Matrix

### For PDF-to-Markdown specifically:

| Priority | Best Choice | Why |
|----------|-------------|-----|
| **Best quality (self-hosted)** | Marker/Surya | Full pipeline, structured MD, tables, formulas |
| **Best quality (API OK)** | Mistral OCR 3 | Fastest, strong accuracy, $1-2/1k pages |
| **Best for academic papers** | Nougat (legacy) or Marker | LaTeX math preservation |
| **Lowest VRAM** | GOT-OCR 2.0 (580M) | ~4-8 GB, decent quality |
| **No GPU at all** | Tesseract + post-processing | CPU only, Apache 2.0, basic quality |
| **Best license** | Tesseract (Apache 2.0) or GOT-OCR (Apache code) | No restrictions |
| **Cutting edge accuracy** | Chandra (83.1 score) | 9B params, from Marker creator |
| **Speed + self-hosted** | DeepSeek-OCR or LightOn OCR | 4.6-5.5 pages/sec on H100 |

### Cost per 1M pages (H100 @ $2.85/hr):

| Tool | Cost Estimate |
|------|---------------|
| LightOn OCR | ~$141 |
| DeepSeek-OCR | ~$168 |
| PaddleOCR-VL | ~$355 |
| Marker/Surya (batch) | ~$400-600 (estimated) |
| Chandra | ~$605 |
| Mistral OCR 3 (API) | $1,000-2,000 |
| Nougat | ~$10,000+ (very slow) |
| Tesseract (CPU) | ~$500+ (CPU cluster) |

---

## Recommendations for This Project

Given that this project (DeepseekOCR_PDF2Markdown) is building a Rust-based PDF-to-Markdown converter:

1. **Primary benchmark target:** Marker/Surya -- this is what users will compare against. Matching or exceeding Marker's quality at competitive speed would be the gold standard.

2. **API fallback consideration:** Mistral OCR 3 at $1/1k pages (batch) is hard to beat on speed. If the project supports hybrid modes (local + API), this could be a useful integration.

3. **Tesseract as floor:** Any solution should meaningfully beat Tesseract on complex layouts. This is the minimum bar.

4. **Watch list:** Chandra (from the same creator as Surya/Marker) and DeepSeek-OCR represent where the field is heading -- VLM-based models that process entire pages as images.

5. **License awareness:** If targeting commercial use, be cautious of Surya's weight restrictions (CC-BY-NC effectively for larger companies), Nougat's CC-BY-NC weights, and GOT-OCR's CC-BY-NC training data. Tesseract and GOT-OCR code (Apache 2.0) are the safest for unrestricted commercial use.

---

## Sources

- [Surya OCR GitHub](https://github.com/datalab-to/surya)
- [Marker GitHub](https://github.com/datalab-to/marker)
- [Nougat GitHub](https://github.com/facebookresearch/nougat)
- [Nougat Paper (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [GOT-OCR 2.0 GitHub](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
- [GOT-OCR Paper (arXiv:2409.01704)](https://arxiv.org/abs/2409.01704)
- [Mistral OCR 3 Announcement](https://mistral.ai/news/mistral-ocr-3)
- [Mistral OCR Technical Review (PyImageSearch)](https://pyimagesearch.com/2025/12/23/mistral-ocr-3-technical-review-sota-document-parsing-at-commodity-pricing/)
- [Tesseract OCR GitHub](https://github.com/tesseract-ocr/tesseract)
- [E2E Networks: 7 Best Open-Source OCR Models 2025](https://www.e2enetworks.com/blog/complete-guide-open-source-ocr-models-2025)
- [Modal: 8 Top Open-Source OCR Models Compared](https://modal.com/blog/8-top-open-source-ocr-models-compared)
- [IntuitionLabs: AI OCR Models Comparison](https://intuitionlabs.ai/articles/ai-ocr-models-pdf-structured-text-comparison)
- [OmniAI OCR Benchmark](https://getomni.ai/blog/ocr-benchmark)
- [Unstract: Best Open-Source OCR Tools 2025](https://unstract.com/blog/best-opensource-ocr-tools-in-2025/)
- [KDnuggets: 10 Awesome OCR Models 2025](https://www.kdnuggets.com/10-awesome-ocr-models-for-2025)
- [Chandra OCR Benchmarks (Skywork)](https://skywork.ai/blog/sheets/chandra-ocr-benchmark/)
- [Deep Dive: PDF to Markdown Open Source (Jimmy Song)](https://jimmysong.io/blog/pdf-to-markdown-open-source-deep-dive/)
