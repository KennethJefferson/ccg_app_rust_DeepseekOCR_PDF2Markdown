# Troubleshooting: CUDA & Marker Failures in Production

## Environment
- **GPU**: RTX 3090 (24 GB VRAM)
- **Server**: 4 uvicorn workers, each with a Marker PdfConverter instance
- **Stack**: marker-pdf 1.10.2, surya-ocr 0.17.1, PyTorch 2.4.1+cu124
- **VRAM per worker**: ~3.5 GB idle, 4-9 GB during processing (varies by PDF complexity)

---

## Failure Mode 1: CUDA Device-Side Assert Triggered

### Symptom
```
CUDA error: device-side assert triggered
Search for 'cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/...
```

### Root Cause
A CUDA kernel encounters an invalid condition (out-of-bounds index, NaN propagation, tensor shape mismatch). In Marker, this typically happens in surya's OCR or layout detection models when processing PDFs with unusual layouts, embedded fonts, or malformed page structures.

### Why It's Fatal
Once a device-side assert fires, **the entire CUDA context for that process is permanently corrupted**. Every subsequent CUDA operation in that process will fail. The worker is a zombie - alive but unable to do GPU work.

### Why Our Code Made It Worse
Our exception handler in `main.py` catches the error and returns it as a response:
```python
except Exception as e:
    return ConvertResponse(success=False, error=str(e))
```
This keeps the worker alive with a corrupted CUDA context, leaking VRAM and failing all future requests routed to it.

### Fix
Detect CUDA errors and let the worker die so uvicorn auto-restarts it with a fresh CUDA context:
```python
except Exception as e:
    if "CUDA" in str(e) or "device-side assert" in str(e):
        logger.critical(f"Fatal CUDA error, worker will exit: {e}")
        os._exit(1)  # Hard exit, uvicorn restarts the worker
    return ConvertResponse(success=False, error=str(e))
```

### Sources
- [PyTorch: How to fix device-side assert](https://discuss.pytorch.org/t/how-to-fix-cuda-error-device-side-assert-triggered-error/137553)
- [PyTorch Issue #17425: Improved assert messages](https://github.com/pytorch/pytorch/issues/17425)
- [Practical tips for device-side assert](https://discuss.pytorch.org/t/practical-tips-for-runtimeerror-cuda-error-device-side-assert-triggered/157167)

---

## Failure Mode 2: CUDA Out of Memory (OOM)

### Symptom
```
CUDA out of memory. Tried to allocate 880.00 MiB. GPU 0 has a total capacity of 23.57 GiB
of which 45.12 MiB is free. Process 663168 has 9.48 GiB memory in use...
```

### Root Cause
4 workers x ~6 GB average during processing = 24 GB = card limit. Large PDFs (300+ pages with complex layouts) cause individual workers to spike to 7-9 GB, pushing total past 24 GB.

### Contributing Factors
- **VRAM fragmentation**: PyTorch's default allocator creates small unusable gaps between allocations
- **Zombie workers**: Workers with corrupted CUDA contexts hold VRAM but can't use it
- **No per-process memory limits**: A single worker processing a huge PDF can starve others

### Fix: PYTORCH_ALLOC_CONF
```bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
```
This changes PyTorch's memory allocator to expand existing segments instead of creating new ones, reducing fragmentation. Should be set in `start.sh` before launching uvicorn.

### Fix: Reduce Workers for Large Batches
With 24 GB, safe configurations:
- **4 workers**: Works for small/medium PDFs (<200 pages). Risk of OOM on large PDFs.
- **3 workers**: Safer. ~7.5 GB headroom per worker. Recommended for mixed workloads.
- **2 workers**: Very safe. ~10 GB per worker. For very large PDFs (400+ pages).

### Fix: Worker Self-Kill on OOM
Similar to device-side assert - OOM often leaves CUDA context in a bad state. Worker should exit and be restarted.

### Sources
- [PyTorch CUDA Memory Management](https://docs.pytorch.org/docs/stable/notes/cuda.html)
- [Understanding CUDA Memory Usage](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html)

---

## Failure Mode 3: Marker Bug - 'NoneType' input_ids

### Symptom
```
'NoneType' object has no attribute 'input_ids'
```

### Root Cause
A Marker/surya internal model returns None where a tokenized output was expected. This happens with specific PDF content that produces empty or unparseable text regions. The model's tokenizer receives no input and returns None.

### Fix
This is a Marker bug - not fixable on our end. We should:
1. Catch and report clearly (already done via success=false)
2. NOT kill the worker (this is a software bug, not CUDA corruption)
3. Consider filing an issue on [datalab-to/marker](https://github.com/datalab-to/marker/issues)

### Related Issues
- [Marker #548: NoneType 'get' error](https://github.com/datalab-to/marker/issues/548)

---

## Failure Mode 4: PDFium Data Format Error

### Symptom
```
Failed to load document (PDFium: Data format error).
```

### Root Cause
The PDF file itself is incompatible with PDFium (Marker's PDF renderer). Causes include:
- Encrypted/DRM-protected PDFs
- PDFs created with non-standard tools
- Corrupted PDF structure
- PDF version incompatibility

### Verification
This error occurs even on localhost (confirmed by testing), ruling out WAN corruption.

### Fix
Not fixable - these PDFs need to be converted with a different tool (e.g., MinerU, Docling) or re-exported from the original source. Report clearly to the user.

---

## Failure Mode 5: Request Timeout (300s)

### Symptom
```
Request timed out after 300s
```

### Root Cause
The 300s client timeout covers the entire round-trip: WAN upload + server processing + response download. For large PDFs:
- Upload 28 MB at ~1 Mbps = 224 seconds
- Processing 400 pages at 0.5-1s/page = 200-400 seconds
- Total: 424-624 seconds > 300s timeout

### Fix
Increase client timeout to 600s or make it dynamic based on file size:
```rust
// Base: 120s for processing + 1s per MB for upload + 1s per MB for download
let timeout_secs = 120 + (file_size_mb * 2) as u64;
```

---

## Server Hardening Checklist

1. **PYTORCH_ALLOC_CONF=expandable_segments:True** - Reduce VRAM fragmentation
2. **Worker self-kill on CUDA errors** - Let uvicorn restart with fresh context
3. **torch.cuda.empty_cache() in exception handler** - Clean up even on failures
4. **Temp file cleanup** - Periodic cleanup of /tmp/marker_* orphans
5. **CUDA_LAUNCH_BLOCKING=1** (debug only) - Better error traces but kills perf
6. **Consider 3 workers** - More headroom per worker, fewer OOM cascades

## Uvicorn Auto-Restart Behavior

When using `uvicorn --workers N`, the master process monitors children and **automatically restarts any worker that exits**. This is key to our recovery strategy: if a worker hits a fatal CUDA error and calls `os._exit(1)`, uvicorn spawns a fresh worker that loads the model and gets a clean CUDA context. No manual intervention needed.

Source: [Uvicorn Deployment Docs](https://www.uvicorn.org/deployment/), [Uvicorn Process Management](https://deepwiki.com/encode/uvicorn/4.2-process-management)
