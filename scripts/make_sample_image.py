"""Generate a synthetic document-like image to sanity-check OCR without needing an external file."""
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 900, 500
OUT_PATH = "samples/sample_doc.png"

LINES = [
    "INVOICE #A-10492",
    "",
    "Bill To: Runto me.",
    "Date: 2026-08-03",
    "",
    "Item                Qty    Price",
    "Widget Pro           3     $19.99",
    "Gadget Mini           1     $49.50",
    "",
    "Total: $109.47",
    "",
    "Thank you for your business!",
]


def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()

    y = 30
    for line in LINES:
        draw.text((40, y), line, fill="black", font=font)
        y += 38

    img.save(OUT_PATH)
    print(f"Saved sample image to {OUT_PATH}")


if __name__ == "__main__":
    main()
