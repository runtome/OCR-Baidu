# OCR-Baidu

Local test harness for [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR), a 3B-parameter
vision-language OCR model, run via Hugging Face Transformers on a local NVIDIA GPU.

## Layout

- `scripts/test_ocr.py` — loads `baidu/Unlimited-OCR` and runs `.infer()` (gundam mode) on an image
- `scripts/test_ocr_pdf.py` — converts a PDF to page images (PyMuPDF) and OCRs pages one at a time
  with `.infer()`, with tunable speed/fidelity settings (see below)
- `scripts/make_sample_image.py` — generates a synthetic sample invoice image for a quick sanity check
- `samples/` — test images
- `outputs/` — OCR results (`result.md` + `result_with_boxes.jpg`) written by `test_ocr.py`
- `Unlimited-OCR/` — upstream GitHub repo, cloned for reference (its `infer.py` targets SGLang, not used here)
- `requirements.txt` — pinned, verified-working dependency set for this setup

## Requirements

- NVIDIA GPU with CUDA support (tested on an RTX 3060 Ti, 8GB VRAM)
- Python 3.11 (the repo's custom model code is not yet compatible with newer Python/transformers releases)
- [`uv`](https://docs.astral.sh/uv/) for environment management

## Setup

```powershell
uv venv .venv --python 3.11
uv pip install -r requirements.txt
```

`requirements.txt` pins `transformers==4.57.1` deliberately — transformers 5.x changed internal APIs
that the model's `trust_remote_code` modeling files rely on, and torch/torchvision are pulled from
PyTorch's `cu128` wheel index for CUDA support.

## Usage

```powershell
uv run python scripts/make_sample_image.py        # optional: generate samples/sample_doc.png
uv run python scripts/test_ocr.py samples/sample_doc.png
```

The first run downloads the model weights from Hugging Face (~6GB) and caches them under
`~/.cache/huggingface/hub`; subsequent runs reuse the cache. Results are written to `outputs/`.

To OCR your own image, pass its path instead:

```powershell
uv run python scripts/test_ocr.py path\to\your\image.jpg
```

### PDF input

```powershell
uv run python scripts/test_ocr_pdf.py path\to\your\document.pdf
```

This renders each page to a PNG (via PyMuPDF, 300 dpi) and OCRs pages **one at a time** in gundam mode
(`model.infer_multi()` packs every page into a single context window, which only works for a handful of
pages before exceeding `max_length=32768` — a 248-page PDF needs ~68k tokens, so per-page looping is
the only approach that scales). Results land in `outputs/<pdf-name>/page_0001/`, `page_0002/`, etc., and
are concatenated into `outputs/<pdf-name>/combined.md`.

Already-processed pages are skipped on rerun, so an interrupted run can just be restarted. For a subset
of pages:

```powershell
uv run python scripts/test_ocr_pdf.py path\to\your\document.pdf --start 1 --end 10
```

### Speed / fidelity tuning

Two flags trade OCR fidelity for speed:

- `--mode {gundam,base}` — `gundam` (default) tiles each page into a grid for higher fidelity on dense/
  small text, at roughly **8-10x the image-token cost** of `base` (a single fixed-size image, no tiling).
  `base` is much faster per page but loses resolution on fine print.
- `--dpi N` (default 300) — lower DPI shrinks the tile grid `gundam` mode picks, cutting its token cost
  independently of `--mode`, at some risk to fidelity on small text.
- `--attn-implementation {eager,flash_attention_2}` (default `eager`) — `flash_attention_2` is optional/
  experimental (the package is notoriously hard to build natively on Windows); the script automatically
  falls back to `eager` if it fails to load.

Non-default `--mode`/`--dpi` combinations get their own output namespace (e.g.
`outputs/<pdf-name>/base_dpi300/`) so a benchmark run never collides with — or silently mixes results
into — a run under different settings. A `settings.json` in each output directory is checked on every
run and aborts loudly if the requested settings don't match what's already there. Use `--tag <name>` to
force a custom namespace.

Each run also writes `timings.json` (per-page seconds) and prints an end-of-run summary (avg/median/min/
max sec-per-page, extrapolated full-document time), so you can measure the actual speed difference
instead of guessing. Recommended first step before committing to a full run — benchmark a small range
across a couple of settings and compare:

```powershell
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

- `test_ocr.py` always uses "gundam" mode (`base_size=1024, image_size=640, crop_mode=True`) — fits
  comfortably in 8GB VRAM. `test_ocr_pdf.py` supports both gundam and base mode (see above).
- `model.infer_multi()` (true multi-page/PDF mode in `Unlimited-OCR/README.md`) only supports "base" mode
  and is meant for a few pages that share context (e.g. a multi-page form), not whole documents.
