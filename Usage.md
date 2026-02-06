# Usage Guide

## CLI Reference

```
pdf2md [OPTIONS] --input <DIR>... --api-url <URL>
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input <DIR>...` | One or more input directories containing PDF files |
| `--api-url <URL>` | Server URL (e.g., `http://<ip>:<port>` or `https://<pod-id>-8000.proxy.runpod.net`) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --recursive` | `false` | Scan subdirectories recursively |
| `-o, --output <DIR>` | next to source | Flat output directory for all `.md` files |
| `-w, --workers <N>` | `2` | Number of parallel upload workers (1-4) |

## Examples

### Basic: Convert PDFs in a single directory

```bash
pdf2md -i ./pdfs --api-url https://abc123-8000.proxy.runpod.net
```

Output `.md` files are written next to each source PDF.

### Recursive scan with multiple input directories

```bash
pdf2md -i ./reports ./invoices -r --api-url https://abc123-8000.proxy.runpod.net
```

### Flat output directory with parallel workers

```bash
pdf2md -i ./documents -r -o ./markdown -w 4 --api-url https://abc123-8000.proxy.runpod.net
```

All output goes into `./markdown/`. If multiple PDFs share the same filename stem, they are auto-renamed:
- `report.md`
- `report_1.md`
- `report_2.md`

### Skip already-converted files

The tool automatically checks if the target `.md` file exists before queuing a PDF. Re-running the same command safely skips completed files:

```bash
# First run: converts 50 PDFs
pdf2md -i ./docs -r --api-url https://abc123-8000.proxy.runpod.net

# Second run: skips the 50 already-converted, only processes new ones
pdf2md -i ./docs -r --api-url https://abc123-8000.proxy.runpod.net
```

## Server Deployment

### On Runpod

1. Create a GPU pod (RTX 3090 / A5000 / A6000 with 24GB+ VRAM)
2. Set **Volume Mount Path** to `/workspace` for persistent storage
3. Expose **TCP port 8000** in the pod config for direct API access
4. Upload the `server/` directory to `/workspace/pdf2md-server/`
5. Run the startup script:

```bash
cd /workspace/pdf2md-server
chmod +x start.sh
./start.sh
```

6. First run creates a venv at `/workspace/venv/`, installs deps, and downloads Marker models to `/root/.cache/datalab/` (~3.3GB). Subsequent starts skip these steps. For persistence across pod restarts, symlink the cache to `/workspace/`: `mv /root/.cache/datalab /workspace/datalab_cache && ln -s /workspace/datalab_cache /root/.cache/datalab`
7. The server listens on port 8000. Use the direct TCP port (e.g., `213.192.x.x:40016`) for fastest access, or the Runpod proxy URL.

### Server Configuration

The server loads multiple Marker model instances for parallel processing. Control the pool size via environment variable:

```bash
# Default: 4 instances (requires ~12-18 GB VRAM)
./start.sh

# Reduce if VRAM is tight
MARKER_POOL_SIZE=2 ./start.sh
```

### Health Check

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
# {"status":"ok","model_loaded":true,"gpu_available":true,"pool_size":4}
```

### Manual Test

```bash
curl -F "file=@test.pdf" https://<pod-id>-8000.proxy.runpod.net/convert
# {"success":true,"markdown":"# Document Title\n\n...","pages_processed":3,"error":null}
```

## Shutdown Behavior

| Action | Effect |
|--------|--------|
| First `Ctrl+C` | Graceful shutdown: workers finish their current PDF, then stop |
| Second `Ctrl+C` | Force quit: immediate exit |

The terminal is always restored to its normal state on exit, even if the program panics.

## Logging

When the TUI is active, logs are written to a daily rolling file in the output directory (or current directory if no `-o` specified):

```
pdf2md.log.2026-02-05
```

Set the log level via the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug pdf2md -i ./docs --api-url ...
```

## Performance

Using Marker on RTX 3090. Throughput scales with pool size and worker count:

| Setup | Throughput (est.) |
|-------|------------------|
| 1 worker / 1 instance | ~150 pages/min |
| 2 workers / 2 instances | ~280 pages/min |
| 4 workers / 4 instances | ~500 pages/min (GPU-bound) |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot reach API server` | Check server is running and URL is correct. Run health check with curl. |
| `Request timed out after 300s` | Large PDFs may exceed the 5-minute timeout. The tool retries up to 3 times automatically. |
| `GPU out of memory` | Reduce pool size with `MARKER_POOL_SIZE=2` and restart, or restart the pod to clear VRAM. |
| `0 PDFs found` | Check input directory path and ensure PDFs have `.pdf` extension. Use `-r` for nested directories. |
| Server won't start after crash | GPU memory may be held by zombie processes. Restart the Runpod pod to clear VRAM. |
| Stale code after SCP update | Delete `app/__pycache__/` and restart uvicorn. |
