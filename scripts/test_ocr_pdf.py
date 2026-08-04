"""OCR a (possibly large) PDF with baidu/Unlimited-OCR via Hugging Face Transformers.

Renders each page to an image with PyMuPDF, then OCRs pages one at a time with
`model.infer()`. Pages are processed independently — `infer_multi()` concatenates
ALL pages into a single context window, which only works for a handful of pages
before blowing past max_length (a 248-page PDF needs ~68k tokens, well past the
32k limit).

Per-page results land in outputs/<pdf-name>[/<settings>]/page_0001/, etc. and are
also concatenated into .../combined.md. Already-done pages are skipped on rerun,
so an interrupted run can just be restarted.

Speed knobs (--mode, --dpi, --attn-implementation):
  --mode gundam (default) tiles each page into a grid (crop_mode=True) for higher
  fidelity on dense/small text, at ~8-10x the image-token cost of --mode base
  (a single fixed-size image, crop_mode=False). --dpi controls how large a grid
  gundam mode picks (lower dpi -> smaller grid -> faster, some fidelity risk).
  --attn-implementation flash_attention_2 is optional/experimental on Windows
  (falls back to eager automatically if it fails to load).

Non-goals for this pass: real multi-page batching (infer() hard-codes batch
size 1; true batching means bypassing infer() to hand-build padded tensors and
call generate() directly - complex, uncertain payoff on 8GB VRAM) and skipping
box-drawing post-processing via eval_mode (real but minor cost; skipping it
means reimplementing infer()'s <|det|>/<|ref|> cleanup ourselves).

Usage:
    uv run python scripts/test_ocr_pdf.py <path-to-pdf> [--start N] [--end N] [--dpi N]
                                           [--mode {gundam,base}] [--attn-implementation {eager,flash_attention_2}]

Recommended first step - benchmark a small range before committing to a full run:
    uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode gundam --dpi 300   # baseline
    uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode gundam --dpi 150   # cheaper tiling
    uv run python scripts/test_ocr_pdf.py doc.pdf --start 38 --end 42 --mode base   --dpi 300   # no tiling
Then compare each run's timings.json (avg sec/page) and spot-check result.md /
result_with_boxes.jpg for accuracy loss before applying new settings document-wide.
"""
import argparse
import json
import os
import statistics
import tempfile
import time

import fitz  # PyMuPDF
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "baidu/Unlimited-OCR"
OUTPUT_DIR = "outputs"
PDF_DPI = 300
PROMPT = "<image>document parsing."
MAX_LENGTH = 32768

DEFAULT_MODE = "gundam"
MODE_PARAMS = {
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),
}


def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> list[str]:
    doc = fitz.open(pdf_path)
    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(tqdm(doc, desc="Rendering pages", unit="page")):
        out = os.path.join(tmp_dir, f"page_{i + 1:04d}.png")
        page.get_pixmap(matrix=mat).save(out)
        paths.append(out)
    doc.close()
    return paths


def resolve_run_subdir(mode: str, dpi: int, tag: str | None) -> str:
    """Returns '' for today's implicit defaults (preserves the existing flat
    outputs/<stem>/page_XXXX/ layout so in-progress runs keep resuming), or a
    settings-derived subdir name otherwise."""
    if tag:
        return tag
    if mode == DEFAULT_MODE and dpi == PDF_DPI:
        return ""
    return f"{mode}_dpi{dpi}"


def check_or_write_settings(doc_output_dir: str, settings: dict) -> None:
    """Guards against silently mixing results produced under different
    OCR-affecting settings inside the same output directory."""
    settings_path = os.path.join(doc_output_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != settings:
            raise SystemExit(
                f"ERROR: {settings_path} was produced with different settings.\n"
                f"  existing:   {existing}\n"
                f"  requested:  {settings}\n"
                f"Pass --tag <name> to use a separate output namespace, or remove "
                f"{doc_output_dir} if you intend to redo it with the new settings."
            )
    else:
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)


def load_model(attn_implementation: str):
    common = dict(trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16)
    if attn_implementation == "flash_attention_2":
        try:
            model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="flash_attention_2", **common)
            print("  attn_implementation=flash_attention_2 loaded OK")
            return model
        except Exception as e:
            print(f"  WARNING: flash_attention_2 unavailable ({e!r}); falling back to eager.")
    model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager", **common)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("--start", type=int, default=1, help="1-based first page (default: 1)")
    parser.add_argument("--end", type=int, default=None, help="1-based last page, inclusive (default: last page)")
    parser.add_argument("--dpi", type=int, default=PDF_DPI)
    parser.add_argument("--mode", choices=["gundam", "base"], default=DEFAULT_MODE,
                         help="gundam = tiled high-res crop_mode (~2300-2700 image tokens/page); "
                              "base = single fixed-size image, no tiling (~273 image tokens/page, faster)")
    parser.add_argument("--attn-implementation", choices=["eager", "flash_attention_2"], default="eager",
                         help="flash_attention_2 is optional/experimental on Windows; "
                              "falls back to eager automatically if it fails to load")
    parser.add_argument("--tag", default=None,
                         help="Explicit output-namespace override, for A/B testing settings "
                              "without recomputing pages under the default layout")
    args = parser.parse_args()
    run_t0 = time.perf_counter()

    pdf_stem = os.path.splitext(os.path.basename(args.pdf_path))[0]
    subdir = resolve_run_subdir(args.mode, args.dpi, args.tag)
    doc_output_dir = os.path.join(OUTPUT_DIR, pdf_stem, subdir) if subdir else os.path.join(OUTPUT_DIR, pdf_stem)
    os.makedirs(doc_output_dir, exist_ok=True)
    combined_path = os.path.join(doc_output_dir, "combined.md")
    timings_path = os.path.join(doc_output_dir, "timings.json")

    mode_params = MODE_PARAMS[args.mode]
    check_or_write_settings(doc_output_dir, {
        "mode": args.mode,
        "dpi": args.dpi,
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        **mode_params,
    })

    print(f"Converting {args.pdf_path} to page images ({args.dpi} dpi)...")
    image_files = pdf_to_images(args.pdf_path, dpi=args.dpi)
    total_pages = len(image_files)
    end = args.end or total_pages
    print(f"  {total_pages} page(s) rendered; processing pages {args.start}-{end} "
          f"(mode={args.mode}, output: {doc_output_dir})")

    print(f"Loading {MODEL_NAME} (first run downloads weights from Hugging Face)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = load_model(args.attn_implementation)
    model = model.eval().cuda()
    print(f"  model load + setup: {time.perf_counter() - run_t0:.1f}s")

    page_timings: list[tuple[int, float]] = []

    for page_num in range(args.start, end + 1):
        image_path = image_files[page_num - 1]
        page_dir = os.path.join(doc_output_dir, f"page_{page_num:04d}")
        result_md = os.path.join(page_dir, "result.md")

        if os.path.exists(result_md):
            print(f"[{page_num}/{end}] already done, skipping")
            continue

        print(f"[{page_num}/{end}] OCR-ing {image_path}...")
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=image_path,
            output_path=page_dir,
            max_length=MAX_LENGTH,
            no_repeat_ngram_size=35,
            ngram_window=128,
            save_results=True,
            **mode_params,
        )
        elapsed = time.perf_counter() - t0
        page_timings.append((page_num, elapsed))
        peak_mem = f", peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f}GB" if torch.cuda.is_available() else ""
        print(f"[{page_num}/{end}] done in {elapsed:.1f}s{peak_mem}")

    print("\nConcatenating per-page results...")
    with open(combined_path, "w", encoding="utf-8") as combined:
        for page_num in range(args.start, end + 1):
            page_dir = os.path.join(doc_output_dir, f"page_{page_num:04d}")
            result_md = os.path.join(page_dir, "result.md")
            combined.write(f"\n\n## Page {page_num}\n\n")
            if os.path.exists(result_md):
                with open(result_md, encoding="utf-8") as f:
                    combined.write(f.read())
            else:
                combined.write("*(missing — page was not processed)*")

    if page_timings:
        durs = [d for _, d in page_timings]
        print("\n--- Timing summary ---")
        print(f"  pages processed this run: {len(durs)}")
        print(f"  total OCR time: {sum(durs):.1f}s")
        print(f"  avg/median/min/max sec/page: "
              f"{statistics.mean(durs):.1f} / {statistics.median(durs):.1f} / {min(durs):.1f} / {max(durs):.1f}")
        if total_pages > len(durs):
            print(f"  extrapolated full-document ({total_pages} pages) time at this avg: "
                  f"{statistics.mean(durs) * total_pages / 60:.1f} min")

        existing_timings = {}
        if os.path.exists(timings_path):
            with open(timings_path, encoding="utf-8") as f:
                existing_timings = json.load(f)
        existing_timings.update({str(p): d for p, d in page_timings})
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(existing_timings, f, indent=2)

    print(f"\nDone in {time.perf_counter() - run_t0:.1f}s total. Combined result: {combined_path}")


if __name__ == "__main__":
    main()
