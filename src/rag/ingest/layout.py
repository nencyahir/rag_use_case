"""Plain-prose PDFs (use case 1): put two-column pages in the order a human reads them."""
import re

import pymupdf

from .textfix import Fixer

FOOTER_Y = 744          # running footers sit below this line on every page of the GPO report
HEADER_Y = 45


def reading_order(blocks: list[dict]) -> list[dict]:
    """Wide blocks (titles, banners, tables) split the page into bands.
    Inside each band: the left column top to bottom, then the right column."""
    ordered, band = [], []

    def flush():
        left = sorted([b for b in band if b["left"]], key=lambda b: b["y0"])
        right = sorted([b for b in band if not b["left"]], key=lambda b: b["y0"])
        ordered.extend(left + right)
        band.clear()

    for b in sorted(blocks, key=lambda b: (b["y0"], b["x0"])):
        if b["wide"]:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def _page_blocks(page) -> list[dict]:
    mid, width = page.rect.width / 2, page.rect.width
    blocks = []
    for x0, y0, x1, y1, text, _n, kind in page.get_text("blocks"):
        if kind != 0 or not text.strip() or y0 > FOOTER_Y or y1 < HEADER_Y:
            continue
        blocks.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text,
                       "wide": (x1 - x0) > width * 0.62 or (x0 < mid - 40 and x1 > mid + 40),
                       "left": x1 <= mid + 25})
    return blocks


_PAGE_NOISE = re.compile(r"^(?:\d+\s+)?(?:2 0 1 8 A N N U A L|U \. S \. G O V E R N|U\.S\. GOVERNMENT PUBLISHING OFFICE \|)", re.I)


def read_pages(path) -> list[dict]:
    """-> [{'page': 1-based number, 'text': clean running text}] in reading order."""
    doc = pymupdf.open(path)
    raw = []
    for number, page in enumerate(doc, start=1):
        parts = []
        for b in reading_order(_page_blocks(page)):
            lines = [l for l in b["text"].split("\n") if l.strip()]
            if lines and _PAGE_NOISE.match(lines[0].strip()):
                continue
            parts.append(lines)
        raw.append((number, parts))
    fixer = Fixer(" ".join(l for _, parts in raw for p in parts for l in p))
    pages = []
    for number, parts in raw:
        text = " ".join(fixer.join(p) for p in parts)
        text = fixer.clean(text, th=False)
        if text.strip():
            pages.append({"page": number, "text": text.strip()})
    return pages
