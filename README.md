# OCR-Baidu

Local test harness for [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), a 3B-parameter
vision-language OCR model, run via Hugging Face Transformers on a local NVIDIA GPU. Works on both
Windows and Linux — see [Platform notes](#platform-notes) for the handful of places commands differ.

## Layout

- `scripts/test_ocr.py` — loads `baidu/Unlimited-OCR` and runs `.infer()` on a single image, or loops
  over a folder of images (one `.md` per image plus a combined `.md`) — see [Folder input](#folder-input)
- `scripts/test_ocr_pdf.py` — converts a PDF to page images (PyMuPDF) and OCRs pages one at a time
  with `.infer()`, with tunable speed/fidelity settings (see below)
- `scripts/make_sample_image.py` — generates a synthetic sample invoice image for a quick sanity check
- `samples/` — test images
- `inputs/` — place a folder of images here for batch OCR (e.g. `inputs/images/`)
- `outputs/` — OCR results (`result.md` + `result_with_boxes.jpg`, or `<folder-name>.md` in folder mode)
- `Unlimited-OCR/` — upstream GitHub repo, cloned for reference (its `infer.py` targets SGLang, not used here)
- `requirements.txt` — pinned, verified-working dependency set for this setup

## Requirements

- NVIDIA GPU with CUDA support (tested on Windows with an RTX 3060 Ti, 8GB VRAM; Linux works the same
  way given a CUDA-capable driver). **CPU-only is not supported** — the model's own `trust_remote_code`
  code hardcodes `.cuda()` calls throughout its `infer()`/`infer_multi()` methods, so it errors out
  immediately on a machine without CUDA, regardless of anything in this repo's scripts.
- Python 3.11 (the repo's custom model code is not yet compatible with newer Python/transformers releases)
- [`uv`](https://docs.astral.sh/uv/) for environment management

## Setup

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

`requirements.txt` pins `transformers==4.57.1` deliberately — transformers 5.x changed internal APIs
that the model's `trust_remote_code` modeling files rely on, and torch/torchvision are pulled from
PyTorch's `cu128` wheel index for CUDA support. If your GPU/driver needs a different CUDA version, swap
the `--extra-index-url` line in `requirements.txt` for the matching index from
[pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

```bash
uv run python scripts/make_sample_image.py        # optional: generate samples/sample_doc.png
uv run python scripts/test_ocr.py samples/sample_doc.png
```

The first run downloads the model weights from Hugging Face (~6GB) and caches them under
`~/.cache/huggingface/hub` (Linux/macOS) or `%USERPROFILE%\.cache\huggingface\hub` (Windows);
subsequent runs reuse the cache. Results are written to `outputs/`.

To OCR your own image, pass its path instead:

```bash
uv run python scripts/test_ocr.py path/to/your/image.jpg
```

### Folder input

```bash
uv run python scripts/test_ocr.py inputs/images
```

OCRs every image in the folder (non-recursive; `.png`/`.jpg`/`.jpeg`/`.bmp`/`.tiff`/`.webp`, alphabetical
order) one at a time. Per-image results land in `outputs/<folder-name>/<image-stem>.md`, and all of them
are concatenated into `outputs/<folder-name>/<folder-name>.md`. Already-processed images are skipped on
rerun. For a subset:

```bash
uv run python scripts/test_ocr.py inputs/images --start 1 --end 10
```

`--mode`/`--attn-implementation`/`--tag` (see [Speed / fidelity tuning](#speed--fidelity-tuning)) work
here too — there's no `--dpi` since these are already-rasterized images, not PDF pages rendered at a
chosen resolution.

### PDF input

```bash
uv run python scripts/test_ocr_pdf.py path/to/your/document.pdf
```

This renders each page to a PNG (via PyMuPDF, 300 dpi) and OCRs pages **one at a time** in gundam mode
(`model.infer_multi()` packs every page into a single context window, which only works for a handful of
pages before exceeding `max_length=32768` — a 248-page PDF needs ~68k tokens, so per-page looping is
the only approach that scales). Results land in `outputs/<pdf-name>/page_0001/`, `page_0002/`, etc., and
are concatenated into `outputs/<pdf-name>/combined.md`.

Already-processed pages are skipped on rerun, so an interrupted run can just be restarted. For a subset
of pages:

```bash
uv run python scripts/test_ocr_pdf.py path/to/your/document.pdf --start 1 --end 10
```

### Speed / fidelity tuning

Two flags trade OCR fidelity for speed:

- `--mode {gundam,base}` — `gundam` (default) tiles each page into a grid for higher fidelity on dense/
  small text, at roughly **8-10x the image-token cost** of `base` (a single fixed-size image, no tiling).
  `base` is much faster per page but loses resolution on fine print.
- `--dpi N` (default 300) — lower DPI shrinks the tile grid `gundam` mode picks, cutting its token cost
  independently of `--mode`, at some risk to fidelity on small text.
- `--attn-implementation {eager,flash_attention_2}` (default `eager`) — `flash_attention_2` is optional/
  experimental; the script automatically falls back to `eager` if it fails to load. The `flash-attn`
  package is notoriously hard to build natively on Windows (no guaranteed prebuilt wheel for every
  torch/CUDA/Python combo); it's a much more realistic option on Linux (`uv pip install flash-attn
  --no-build-isolation`, needs a matching CUDA toolkit installed).

Non-default `--mode`/`--dpi` combinations get their own output namespace (e.g.
`outputs/<pdf-name>/base_dpi300/`) so a benchmark run never collides with — or silently mixes results
into — a run under different settings. A `settings.json` in each output directory is checked on every
run and aborts loudly if the requested settings don't match what's already there. Use `--tag <name>` to
force a custom namespace.

Each run also writes `timings.json` (per-page seconds) and prints an end-of-run summary (avg/median/min/
max sec-per-page, extrapolated full-document time), so you can measure the actual speed difference
instead of guessing. Recommended first step before committing to a full run — benchmark a small range
across a couple of settings and compare:

```bash
uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode gundam --dpi 300   # baseline
uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode gundam --dpi 150   # cheaper tiling
uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode base   --dpi 300   # no tiling
```

Then compare `timings.json` across the three directories and spot-check `result.md` /
`result_with_boxes.jpg` for accuracy loss before applying new settings document-wide.

Real multi-page batching (processing several pages in one GPU forward pass) and skipping the box-drawing
post-processing step are both deliberately out of scope — see the module docstring in `test_ocr_pdf.py`
for why.

## Notes

- Both `test_ocr.py` and `test_ocr_pdf.py` default to "gundam" mode (`base_size=1024, image_size=640,
  crop_mode=True`) and support `--mode base` — see [Speed / fidelity tuning](#speed--fidelity-tuning).
  We benchmarked both on a real dense-text document: `base` mode visibly corrupted non-Latin text
  (garbled words, repeated-character artifacts) to save only modest time — `gundam` is the recommended
  default unless you've verified `base` holds up on your specific documents.
- `model.infer_multi()` (true multi-page/PDF mode in `Unlimited-OCR/README.md`) only supports "base" mode
  and is meant for a few pages that share context (e.g. a multi-page form), not whole documents.

## Platform notes

Commands above use `bash` syntax (forward slashes); on Windows they work the same via `uv run python ...`
regardless of shell, but a few things differ:

| | Windows (PowerShell) | Linux / macOS (bash) |
|---|---|---|
| Install `uv` | `winget install astral-sh.uv` or see [docs](https://docs.astral.sh/uv/getting-started/installation/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Activate venv manually (rarely needed — `uv run` does this for you) | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| HF cache location | `%USERPROFILE%\.cache\huggingface\hub` | `~/.cache/huggingface/hub` |
| Path separators in commands | Either `\` or `/` work | `/` only |

Everything else (`uv venv`, `uv pip install`, `uv run python scripts/...`) is identical on both platforms.
GPU driver setup (NVIDIA driver + CUDA-capable card) is a prerequisite on either OS but is out of scope
for this README — see NVIDIA's install docs for your distro if you're on a fresh Linux machine.
