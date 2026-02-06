# Usage Guide

## CLI Reference

```
deepseek-ocr-pdf2md [OPTIONS] --input <DIR>... --api-url <URL>
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i, --input <DIR>...` | One or more input directories containing PDF files |
| `--api-url <URL>` | DeepSeek-OCR server URL (e.g., `https://<pod-id>-8000.proxy.runpod.net`) |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-r, --recursive` | `false` | Scan subdirectories recursively |
| `-o, --output <DIR>` | next to source | Flat output directory for all `.md` files |
| `-w, --workers <N>` | `1` | Number of parallel upload workers |

## Examples

### Basic: Convert PDFs in a single directory

```bash
deepseek-ocr-pdf2md -i ./pdfs --api-url https://abc123-8000.proxy.runpod.net
```

Output `.md` files are written next to each source PDF.

### Recursive scan with multiple input directories

```bash
deepseek-ocr-pdf2md -i ./reports ./invoices -r --api-url https://abc123-8000.proxy.runpod.net
```

### Flat output directory with parallel workers

```bash
deepseek-ocr-pdf2md -i ./documents -r -o ./markdown -w 4 --api-url https://abc123-8000.proxy.runpod.net
```

All output goes into `./markdown/`. If multiple PDFs share the same filename stem, they are auto-renamed:
- `report.md`
- `report_1.md`
- `report_2.md`

### Skip already-converted files

The tool automatically checks if the target `.md` file exists before queuing a PDF. Re-running the same command safely skips completed files:

```bash
# First run: converts 50 PDFs
deepseek-ocr-pdf2md -i ./docs -r --api-url https://abc123-8000.proxy.runpod.net

# Second run: skips the 50 already-converted, only processes new ones
deepseek-ocr-pdf2md -i ./docs -r --api-url https://abc123-8000.proxy.runpod.net
```

## Server Deployment

### On Runpod

1. Create a GPU pod (RTX 3090 / A5000 / A6000 with 24GB+ VRAM)
2. Upload the `server/` directory to `/workspace/deepseek-ocr-server/`
3. Run the startup script:

```bash
cd /workspace/deepseek-ocr-server
chmod +x start.sh
./start.sh
```

4. The first run downloads the model (~6GB). Subsequent starts use the cached model.
5. The server listens on port 8000. Use the Runpod proxy URL for external access.

### Health Check

```bash
curl https://<pod-id>-8000.proxy.runpod.net/health
# {"status":"ok","model_loaded":true,"gpu_available":true}
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
deepseek-ocr.log.2026-02-05
```

Set the log level via the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug deepseek-ocr-pdf2md -i ./docs --api-url ...
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Cannot reach API server` | Check server is running and URL is correct. Run health check with curl. |
| `Request timed out after 300s` | Large PDFs may exceed the 5-minute timeout. The tool retries up to 3 times automatically. |
| `GPU out of memory` | Server returns partial results for completed pages. Reduce PDF page size or restart the server pod. |
| `0 PDFs found` | Check input directory path and ensure PDFs have `.pdf` extension. Use `-r` for nested directories. |
