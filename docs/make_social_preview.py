"""Regenerate the GitHub social preview card.

The card is what a stranger sees before they see the repo — in a LinkedIn
Featured block, a Slack unfurl, or a search result. GitHub's auto-generated
default shows stars and forks, which for a new repo says "nobody has looked at
this". Three measured numbers say something worth clicking.

    python docs/make_social_preview.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 640                    # GitHub's required social-preview size
BG, FG, MUTED = (13, 17, 23), (230, 237, 243), (125, 133, 144)
GREEN, RED, LINE = (63, 185, 80), (248, 81, 73), (48, 54, 61)

OUT = Path(__file__).parent / "social-preview.png"

# Keep in sync with the README. If a number changes there, change it here.
HEADLINE = "How often does RAG retrieval actually return the right page?"
STATS = [
    ("108", "questions, all\nhand-verified", FG),
    ("18", "survived\nverification", GREEN),
    ("33.3%", "BM25 — beat\ndense retrieval", GREEN),
]
FOOTER = "1,338 pages  ·  6,748 chunks  ·  no API key  ·  CI reruns the published eval"
FINDING = "LSA dense retrieval lost at every dimensionality from 64 to 512"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = 72

    d.rectangle([0, 0, 8, H], fill=GREEN)
    d.text((x, 72), "ShubhenduKH", font=font(26), fill=MUTED)
    d.text((x, 116), "rag-retrieval-audit", font=font(64, True), fill=FG)
    d.text((x, 206), HEADLINE, font=font(30), fill=MUTED)
    d.line([(x, 268), (W - 72, 268)], fill=LINE, width=2)

    cx = x
    for value, label, colour in STATS:
        d.text((cx, 308), value, font=font(58, True), fill=colour)
        d.multiline_text((cx, 382), label, font=font(23), fill=MUTED, spacing=8)
        cx += 310

    d.line([(x, 486), (W - 72, 486)], fill=LINE, width=2)
    d.text((x, 514), FOOTER, font=font(24), fill=MUTED)
    d.text((x, 560), FINDING, font=font(24), fill=RED)

    img.save(OUT, optimize=True)
    print(f"wrote {OUT}  {OUT.stat().st_size / 1024:.0f} KB  {img.size}")


if __name__ == "__main__":
    main()
