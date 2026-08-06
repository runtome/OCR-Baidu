"""OCR a single image or a folder of images with baidu/Unlimited-OCR via Hugging Face Transformers.

Single-image usage (unchanged from before, writes flat outputs/result.md):
    uv run python scripts/test_ocr.py samples/sample_doc.png

Folder usage: OCRs every image in the folder one at a time, writing
outputs/<folder_name>/<image_stem>.md per image plus a combined
outputs/<folder_name>/<folder_name>.md. Already-done images are skipped on
rerun, so an interrupted run can just be restarted.
    uv run python scripts/test_ocr.py inputs/images [--start N] [--end N]

Speed knobs (--mode, --attn-implementation) match scripts/test_ocr_pdf.py:
  --mode gundam (default) tiles the image into a grid (crop_mode=True) for
  higher fidelity on dense/small text, at ~8-10x the image-token cost of
  --mode base (a single fixed-size image, crop_mode=False). We benchmarked
  both on this project's real document and found base mode visibly corrupts
  dense non-Latin text (garbled words, repeated-character artifacts) - gundam
  stays the recommended default. --attn-implementation flash_attention_2 is
  optional/experimental on Windows (falls back to eager automatically if it
  fails to load).
"""
import argparse
import json
import os
import shutil
import statistics
import time

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "baidu/Unlimited-OCR"
OUTPUT_DIR = "outputs"
PROMPT = "<image>document parsing."
MAX_LENGTH = 32768
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

DEFAULT_MODE = "gundam"
MODE_PARAMS = {
    "gundam": dict(base_size=1024, image_size=640, crop_mode=True),
    "base": dict(base_size=1024, image_size=1024, crop_mode=False),
}


def discover_images(folder: str) -> list[str]:
    names = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    )
    return [os.path.join(folder, name) for name in names]


def resolve_run_subdir(mode: str, tag: str | None) -> str:
    """Returns '' for today's implicit default (gundam, no tag), preserving
    flat/unnamespaced output so existing usage/docs keep working unchanged."""
    if tag:
        return tag
    if mode == DEFAULT_MODE:
        return ""
    return mode


def check_or_write_settings(output_dir: str, settings: dict) -> None:
    """Guards against silently mixing results produced under different
    OCR-affecting settings inside the same output directory."""
    settings_path = os.path.join(output_dir, "settings.json")
    if os.path.exists(settings_path):
        with open(settings_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != settings:
            raise SystemExit(
                f"ERROR: {settings_path} was produced with different settings.\n"
                f"  existing:   {existing}\n"
                f"  requested:  {settings}\n"
                f"Pass --tag <name> to use a separate output namespace, or remove "
                f"{output_dir} if you intend to redo it with the new settings."
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


def ocr_one(tokenizer, model, image_path: str, output_path: str, mode_params: dict,
            max_length: int, no_repeat_ngram_size: int, ngram_window: int,
            temperature: float) -> tuple[float, float | None]:
    """Runs infer() on one image, returns (elapsed_seconds, peak_vram_gb|None)."""
    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.infer(
        tokenizer,
        prompt=PROMPT,
        image_file=image_path,
        output_path=output_path,
        max_length=max_length,
        no_repeat_ngram_size=no_repeat_ngram_size,
        ngram_window=ngram_window,
        temperature=temperature,
        save_results=True,
        **mode_params,
    )
    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    return elapsed, peak_vram


def run_single_image(args, tokenizer, model, run_t0: float) -> None:
    mode_params = MODE_PARAMS[args.mode]
    subdir = resolve_run_subdir(args.mode, args.tag)
    output_path = os.path.join(OUTPUT_DIR, subdir) if subdir else OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)

    check_or_write_settings(output_path, {
        "mode": args.mode,
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        **mode_params,
    })

    print(f"Running OCR on {args.input_path} (mode={args.mode})...")
    elapsed, peak_vram = ocr_one(tokenizer, model, args.input_path, output_path, mode_params,
                                  args.max_length, args.no_repeat_ngram_size, args.ngram_window,
                                  args.temperature)
    vram_str = f", peak VRAM {peak_vram:.2f}GB" if peak_vram is not None else ""
    print(f"Done in {elapsed:.1f}s{vram_str}")

    result_md = os.path.join(output_path, "result.md")
    if os.path.exists(result_md):
        with open(result_md, encoding="utf-8") as f:
            print("\n--- OCR Result ---")
            print(f.read())

    print(f"\nSaved results under {output_path}/ ({time.perf_counter() - run_t0:.1f}s total)")


def run_folder(args, tokenizer, model, run_t0: float) -> None:
    mode_params = MODE_PARAMS[args.mode]
    folder_name = os.path.basename(os.path.normpath(args.input_path))
    subdir = resolve_run_subdir(args.mode, args.tag)
    output_dir = os.path.join(OUTPUT_DIR, folder_name, subdir) if subdir else os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, f"{folder_name}.md")
    timings_path = os.path.join(output_dir, "timings.json")

    check_or_write_settings(output_dir, {
        "mode": args.mode,
        "prompt": PROMPT,
        "max_length": MAX_LENGTH,
        **mode_params,
    })

    image_files = discover_images(args.input_path)
    total_images = len(image_files)
    end = args.end or total_images
    print(f"  {total_images} image(s) found; processing {args.start}-{end} (mode={args.mode}, output: {output_dir})")

    item_timings: list[tuple[int, float]] = []

    for idx in range(args.start, end + 1):
        image_path = image_files[idx - 1]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        item_dir = os.path.join(output_dir, stem)
        item_result_md = os.path.join(item_dir, "result.md")
        flat_md = os.path.join(output_dir, f"{stem}.md")

        if os.path.exists(item_result_md):
            print(f"[{idx}/{end}] {stem} already done, skipping")
            shutil.copyfile(item_result_md, flat_md)
            continue

        print(f"[{idx}/{end}] OCR-ing {image_path}...")
        elapsed, peak_vram = ocr_one(tokenizer, model, image_path, item_dir, mode_params,
                                      args.max_length, args.no_repeat_ngram_size, args.ngram_window,
                                      args.temperature)
        item_timings.append((idx, elapsed))
        vram_str = f", peak VRAM {peak_vram:.2f}GB" if peak_vram is not None else ""
        print(f"[{idx}/{end}] done in {elapsed:.1f}s{vram_str}")
        shutil.copyfile(item_result_md, flat_md)

    print("\nConcatenating per-image results...")
    with open(combined_path, "w", encoding="utf-8") as combined:
        for idx in range(args.start, end + 1):
            image_path = image_files[idx - 1]
            stem = os.path.splitext(os.path.basename(image_path))[0]
            flat_md = os.path.join(output_dir, f"{stem}.md")
            combined.write(f"\n\n## {stem}\n\n")
            if os.path.exists(flat_md):
                with open(flat_md, encoding="utf-8") as f:
                    combined.write(f.read())
            else:
                combined.write("*(missing — image was not processed)*")

    if item_timings:
        durs = [d for _, d in item_timings]
        print("\n--- Timing summary ---")
        print(f"  images processed this run: {len(durs)}")
        print(f"  total OCR time: {sum(durs):.1f}s")
        print(f"  avg/median/min/max sec/image: "
              f"{statistics.mean(durs):.1f} / {statistics.median(durs):.1f} / {min(durs):.1f} / {max(durs):.1f}")
        if total_images > len(durs):
            print(f"  extrapolated full-folder ({total_images} images) time at this avg: "
                  f"{statistics.mean(durs) * total_images / 60:.1f} min")

        existing_timings = {}
        if os.path.exists(timings_path):
            with open(timings_path, encoding="utf-8") as f:
                existing_timings = json.load(f)
        existing_timings.update({str(i): d for i, d in item_timings})
        with open(timings_path, "w", encoding="utf-8") as f:
            json.dump(existing_timings, f, indent=2)

    print(f"\nDone in {time.perf_counter() - run_t0:.1f}s total. Combined result: {combined_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to a single image, or a folder of images")
    parser.add_argument("--start", type=int, default=1, help="1-based first image index (folder mode only)")
    parser.add_argument("--end", type=int, default=None, help="1-based last image index, inclusive (folder mode only)")
    parser.add_argument("--mode", choices=["gundam", "base"], default=DEFAULT_MODE,
                         help="gundam = tiled high-res crop_mode (higher fidelity, slower); "
                              "base = single fixed-size image, no tiling (faster, lower fidelity)")
    parser.add_argument("--attn-implementation", choices=["eager", "flash_attention_2"], default="eager",
                         help="flash_attention_2 is optional/experimental on Windows; "
                              "falls back to eager automatically if it fails to load")
    parser.add_argument("--tag", default=None,
                         help="Explicit output-namespace override, for A/B testing settings "
                              "without recomputing under the default layout")
    parser.add_argument("--temperature", type=float, default=0.0,
                         help="0.0 (default) = greedy decoding. On unclear/handwritten content, greedy "
                              "decoding can get stuck in a degenerate loop that no-repeat-ngram doesn't "
                              "catch (e.g. a repeated fraction interleaved with an incrementing counter, "
                              "so no fixed-size window repeats verbatim). If a specific image loops, "
                              "retry it with e.g. --temperature 0.2-0.4 to break out via sampling.")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=35,
                         help="Block repeats of this many consecutive tokens (default: 35). Lower it "
                              "(e.g. 10-15) to catch shorter repeating sub-phrases on a looping image.")
    parser.add_argument("--ngram-window", type=int, default=128,
                         help="How far back to look for repeats (default: 128 tokens).")
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH,
                         help=f"Hard cap on total sequence length (prompt+image+output tokens), default "
                              f"{MAX_LENGTH}. Generation stops (possibly mid-result) once hit — use a "
                              f"lower value (e.g. 6000) as a time-bound safety net when retrying a "
                              f"looping image, since a normal page/image needs nowhere near this many.")
    args = parser.parse_args()
    run_t0 = time.perf_counter()

    is_folder = os.path.isdir(args.input_path)
    if not is_folder and (args.start != 1 or args.end is not None):
        parser.error("--start/--end only apply when input_path is a folder")

    print(f"Loading {MODEL_NAME} (first run downloads weights from Hugging Face)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = load_model(args.attn_implementation)
    model = model.eval().cuda()
    print(f"  model load + setup: {time.perf_counter() - run_t0:.1f}s")

    if is_folder:
        run_folder(args, tokenizer, model, run_t0)
    else:
        run_single_image(args, tokenizer, model, run_t0)


if __name__ == "__main__":
    main()
