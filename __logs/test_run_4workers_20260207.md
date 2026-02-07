# Test Run: 4 Workers, 28 PDFs - 2026-02-07

## Configuration
- Server: gunicorn with 4 workers (UvicornWorker), RTX 3090 (24 GB VRAM)
- Client: 4 parallel workers over WAN (213.192.2.119:40016)
- Input: 28 PDFs ranging from 1.1 MB to 67 MB
- Hardening features: max-requests 100, PDF pre-validation, CUDA health probe

## Results
- **Successful conversions: 1/28** (run aborted early due to cascading failures)
- **Run aborted** after ~21 minutes due to OOM cascade and temp file bug

## Timeline

| Time (UTC) | Event |
|---|---|
| 02:55:47 | gunicorn started, 4 workers booting |
| 02:56:28 | All 4 workers ready |
| 03:00:02 | Client connected, health check OK (376ms) |
| 03:00:04 | First uploads arriving (C++ books, C# in Depth) |
| 03:03:37 | C++ Programming Cookbook: 316 pages in 215s (0.68s/page) - SUCCESS |
| 03:04:15 | C++ Software Design: `list index out of range` - Marker bug |
| 03:06:49 | CUDA OOM hit - worker crashed, 2 retries triggered |
| 03:06:50 | gunicorn booting replacement workers (PIDs 96758, 96834) |
| 03:07:35 | Complex Fuzzy Rough Sets: `FileNotFoundError` - temp file deleted by new worker |
| 03:13:02 | Cloud-Native Python: `FileNotFoundError` - same bug |
| 03:17:08 | Calculus Unraveled (67 MB, 684 pages): timeout after 600s |
| 03:21:00 | Run aborted manually |

## VRAM Observations

| Time | VRAM Used | GPU Util | Temp | Power |
|---|---|---|---|---|
| 02:57 (idle) | 14,242 MiB (58%) | 0% | 31C | 25W |
| 03:01 (active) | 20,582 MiB (84%) | 100% | 60C | 297W |
| 03:07 (post-OOM) | 12,979 MiB (53%) | 84% | 68C | 317W |
| 03:14 (active) | 20,168 MiB (82%) | 98% | 66C | 330W |
| 03:17 (peak) | 21,780 MiB (89%) | 100% | 68C | 349W |

## Errors Found

### 1. CUDA OOM (Fatal)
4 workers collectively consumed 23.49 GB (7.35 + 3.47 + 7.05 + 5.62 GB), leaving only 71 MiB free. Worker tried to allocate 338 MiB and OOM'd. Error recovery worked correctly: `os._exit(1)` triggered, gunicorn restarted worker.

### 2. Temp File Cleanup Bug (NEW - Fixed)
**Root cause**: `_cleanup_stale_temp_files()` in the FastAPI lifespan runs `glob.glob("/tmp/marker_*.pdf")` and deletes ALL marker temp files. When gunicorn restarts a crashed worker, the new worker's lifespan deletes temp files that the 3 surviving workers are still using. Marker then throws `FileNotFoundError` when it tries to re-open the PDF during later processing stages (layout recognition, text extraction).

**Impact**: Every OOM crash cascaded into FileNotFoundError for ALL in-flight requests across all workers.

**Fix**: Removed `_cleanup_stale_temp_files()` from lifespan. `start.sh` already cleans temp files before any workers boot, which is safe.

### 3. Marker Bugs (Known)
- `list index out of range` on C++ Software Design PDF - internal Marker bug
- These are per-PDF issues, not recoverable by retry

### 4. Client Timeout (600s)
Calculus Unraveled (67 MB, 684 pages) timed out. At 0.68s/page that PDF would need ~465s processing time, plus upload time over WAN for 67 MB. Likely hit server-side slowdown from VRAM pressure.

## Conclusions
- **4 workers is not viable on RTX 3090**: Peak active VRAM of 21.8 GB leaves < 3 GB headroom, OOM is near-certain with large PDFs
- **3 workers recommended**: ~10.5 GB idle, ~15-18 GB active, leaving 6-9 GB headroom
- **Temp file cleanup bug was critical**: Single OOM crashed ALL in-flight work, not just the affected worker
- **Error recovery (gunicorn restart, client retry) works correctly** for isolated failures
- **Default changed to 3 workers** in start.sh
