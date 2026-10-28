"""Visual brochure PDFs (use case 3): rebuild tables, keep footnote markers, link markers to endnotes."""
import re

import pymupdf

FOOTER_Y = 744
BAR_FILL = (0.23, 0.24, 0.23)            # the dark header bars that sit on top of every table in this PDF
ROW_MAX_SIZE = 6.9                       # table cells are 6.0-6.5pt; surrounding prose is 7pt and up
ENDNOTE_FIRST_PAGE = 11


def line_text(line: dict) -> str:
    """Join the spans of a line. A superscript digit becomes a marker: 'achieved' + '7' -> 'achieved[^7]'."""
    out = ""
    for s in line["spans"]:
        t = s["text"]
        if s["flags"] & 1 and t.strip().isdigit():
            out = out.rstrip() + f"[^{t.strip()}]"
        else:
            out += t
    return out.strip()


def _lines(page):
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            text = line_text(l)
            if text:
                yield {"text": text, "size": l["spans"][0]["size"], "bbox": pymupdf.Rect(l["bbox"]), "block": b["number"]}


def _bars(page):
    return [d["rect"] for d in page.get_drawings()
            if d.get("fill") and all(abs(a - b) < .02 for a, b in zip(d["fill"], BAR_FILL))
            and d["rect"].width > 40 and 8 < d["rect"].height < 40]


def _blocks_of(bars):
    """Bars that touch each other belong to one table header."""
    groups = []
    for r in bars:
        hit = [g for g in groups if any(pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1).intersects(o) for o in g)]
        merged = [r] + [o for g in hit for o in g]
        groups = [g for g in groups if g not in hit] + [merged]
    return groups


def _inside(line_rect, rect, pad=1.5):
    cx, cy = (line_rect.x0 + line_rect.x1) / 2, (line_rect.y0 + line_rect.y1) / 2
    return rect.x0 - pad <= cx <= rect.x1 + pad and rect.y0 - pad <= cy <= rect.y1 + pad


def rebuild_tables(page, number: int) -> list[dict]:
    """Find each table's header bars, then rebuild rows from word positions under them.
    -> [{'text', 'top', 'bottom', 'x0', 'x1'}]"""
    lines = list(_lines(page))
    groups = sorted(_blocks_of(_bars(page)), key=lambda g: min(r.y0 for r in g))
    tables = []
    for gi, group in enumerate(groups):
        union = pymupdf.Rect(min(r.x0 for r in group), min(r.y0 for r in group), max(r.x1 for r in group), max(r.y1 for r in group))
        leaves = sorted([r for r in group if r.y1 >= union.y1 - 1], key=lambda r: r.x0)
        parents = [r for r in group if r.y1 < union.y1 - 1]

        def words_in(rect):
            return " ".join(l["text"] for l in lines if _inside(l["bbox"], rect, 0.5))

        headers = []
        for leaf in leaves:
            parts = [words_in(p) for p in parents if p.x0 - 1 <= (leaf.x0 + leaf.x1) / 2 <= p.x1 + 1]
            leaf_text = words_in(leaf)
            headers.append(" - ".join([x for x in parts + [leaf_text] if x]))
        # where the table body ends: the next table's header, or the first line that is not a table row
        below = [g for g in groups[gi + 1:] if min(r.y0 for r in g) > union.y1]
        limit = min([min(r.y0 for r in g) for g in below] + [FOOTER_Y])
        body = [l for l in lines if l["bbox"].y0 >= union.y1 - 0.5 and l["bbox"].y1 <= limit
                and union.x0 - 6 <= l["bbox"].x0 <= union.x1 + 6]
        body.sort(key=lambda l: (l["bbox"].y0, l["bbox"].x0))
        rows, bottom = [], union.y1
        for l in body:
            if l["size"] >= ROW_MAX_SIZE:
                break                                              # prose below the table
            if rows and abs((l["bbox"].y0 + l["bbox"].y1) / 2 - rows[-1]["y"]) < 4:
                rows[-1]["cells"].append(l)
            else:
                rows.append({"y": (l["bbox"].y0 + l["bbox"].y1) / 2, "cells": [l]})
            bottom = max(bottom, l["bbox"].y1)
        if not rows:
            continue
        out = []
        for r in rows:
            cols = {}
            for c in r["cells"]:
                idx = max([i for i, leaf in enumerate(leaves) if c["bbox"].x0 >= leaf.x0 - 3] or [0])
                cols.setdefault(idx, []).append(c["text"])
            label = " ".join(cols.get(0, []))
            vals = [(headers[i], " ".join(cols[i])) for i in sorted(cols) if i > 0]
            if len(vals) == 1:
                out.append(f"{label}: {vals[0][1]}")
            else:
                out.append(f"{label}: " + "; ".join(f"{h}={v}" for h, v in vals))
        title = headers[0]
        columns = "; ".join(headers[1:])
        text = f"Table: {title}. {'Column' if len(headers) == 2 else 'Columns'}: {columns}\n" + "\n".join(out)
        tables.append({"text": text, "top": union.y0, "bottom": bottom, "x0": union.x0, "x1": union.x1})
    return tables


def read_endnotes(doc) -> dict[int, str]:
    """Endnotes live on the last pages: a number, then text that may wrap over several lines."""
    notes, current = {}, None
    for number in range(ENDNOTE_FIRST_PAGE, len(doc) + 1):
        page = doc[number - 1]
        tables = rebuild_tables(page, number)
        for l in sorted(_lines(page), key=lambda l: (l["bbox"].y0, l["bbox"].x0)):
            y0 = l["bbox"].y0
            if y0 > FOOTER_Y or l["text"].startswith("©") or l["text"] == "Endnotes" or l["size"] >= 20 or (y0 > 700 and l["size"] <= 6.0):
                continue
            if any(t["top"] - 3 <= y0 <= t["bottom"] + 3 and l["bbox"].x0 >= t["x0"] - 6 for t in tables):
                continue
            m = re.match(r"^(?:\[\^(\d{1,2})\]|(\d{1,2})(?=\s))\s*(\S.*)", l["text"]) if l["bbox"].x0 < 213 else None
            if m:
                current = int(m.group(1) or m.group(2))
                notes[current] = m.group(3)
            elif current is not None:
                notes[current] += " " + l["text"]
    return {n: re.sub(r"\s+", " ", t).strip() for n, t in notes.items()}


def _column_order(blocks):
    """Brochure layout: full-width blocks split the page into bands; inside a band read column by column
    (columns found by clustering block centres), so a card's heading stays with its body text."""
    ordered, band = [], []

    def flush():
        if not band:
            return
        centres = sorted({round((b["x0"] + b["x1"]) / 2) for b in band})
        cols, cur = [], [centres[0]]
        for c in centres[1:]:
            if c - cur[-1] > 45:
                cols.append(cur)
                cur = []
            cur.append(c)
        cols.append(cur)

        def col_of(b):
            c = round((b["x0"] + b["x1"]) / 2)
            return next(i for i, group in enumerate(cols) if c in group)
        ordered.extend(sorted(band, key=lambda b: (col_of(b), b["y0"])))
        band.clear()

    for b in sorted(blocks, key=lambda b: (b["y0"], b["x0"])):
        if b["x1"] - b["x0"] > 300:
            flush()
            ordered.append(b)
        else:
            band.append(b)
    flush()
    return ordered


def prose_blocks(page, exclude: list[dict]) -> list[str]:
    """Text blocks in reading order, minus footer, tables and endnotes, with [^n] markers in place."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        texts = []
        for l in b.get("lines", []):
            rect = pymupdf.Rect(l["bbox"])
            if rect.y0 > FOOTER_Y or any(t["top"] - 4 <= rect.y0 <= t["bottom"] + 4 and rect.x0 >= t["x0"] - 6 for t in exclude):
                continue
            t = line_text(l)
            if t:
                texts.append(t)
        if texts:
            blocks.append({"x0": b["bbox"][0], "y0": b["bbox"][1], "x1": b["bbox"][2], "text": " ".join(texts)})
    return [b["text"] for b in _column_order(blocks)]
