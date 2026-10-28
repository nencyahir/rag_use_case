"""Long structured PDFs (use case 2): rebuild the outline (chapter > section > sub-section, region, boxes)
from the fonts, so every passage knows where it lives."""
import re

import pymupdf

from .textfix import Fixer
from ..config import REGIONS

CHAPTERS = {
    "1": "Chapter 1: Global Outlook",
    "2": "Chapter 2: Regional Outlooks",
    "3": "Chapter 3: A Rising Challenge: Sovereign Debt Levels and Interest Rates in EMDEs",
    "4": "Chapter 4: Navigating Volatility: Fiscal Policy and Commodity Price Swings",
}
HEADER_Y, FOOTER_Y = 55, 752


def line_kind(text: str, size: float, font: str, bold: bool) -> str:
    """This report's typography: body is 11pt Garamond; headings are Helvetica."""
    helvetica = "Helvetica" in font
    if re.match(r"^(BOX|TABLE|FIGURE)\s+\d", text) and helvetica and bold:
        return "caption"                                           # "BOX 1.1 How much will AI ..."
    if helvetica and bold and size >= 13.5:
        return "section"                                           # 16pt / 14pt bold: "South Asia", "Global context"
    if helvetica and not bold and abs(size - 11) < 0.3 and len(text) < 90 and not text.endswith("."):
        return "subsection"                                        # "Recent developments", "Outlook"
    if helvetica and "Obl" in font and size >= 7.5 and len(text) > 25:
        return "figtext"                                           # the one-line story under a figure title
    if size < 9.2:
        return "small"                                             # chart labels, table cells, footnotes
    return "body"


def _read_lines(page, fixer_raw=None):
    """All text lines of a page with position/typography, in reading order (bands of left col then right col)."""
    mid, width = page.rect.width / 2, page.rect.width
    boxes = [d["rect"] for d in page.get_drawings()
             if d.get("fill") and d["rect"].width > 150 and d["rect"].height > 40
             and abs(d["fill"][0] - .99) < .02 and abs(d["fill"][1] - .91) < .03]       # the peach box background
    items = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            x0, y0, x1, y1 = l["bbox"]
            if y0 < HEADER_Y or y1 > FOOTER_Y:
                continue
            first = spans[0]
            text = "".join(s["text"] for s in l["spans"]).strip()
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            items.append({"text": text, "size": first["size"], "font": first["font"], "bold": bool(first["flags"] & 16),
                          "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "in_box": any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in boxes),
                          "wide": (x1 - x0) > width * 0.62 or (x0 < mid - 40 and x1 > mid + 40),
                          "left": x1 <= mid + 25})
    ordered, band = [], []

    def flush():
        ordered.extend(sorted([i for i in band if i["left"]], key=lambda i: i["y0"])
                       + sorted([i for i in band if not i["left"]], key=lambda i: i["y0"]))
        band.clear()

    for i in sorted(items, key=lambda i: (i["y0"], i["x0"])):
        if i["wide"] and i["size"] > 9.2:
            flush()
            ordered.append(i)
        else:
            band.append(i)
    flush()
    return ordered


def parse_report(path) -> list[dict]:
    """-> sections: [{chapter, region, box, kind, path, title, lines: [(text, page)]}]"""
    doc = pymupdf.open(path)
    pages = [_read_lines(p) for p in doc]
    fixer = Fixer(" ".join(i["text"] for page in pages for i in page))

    sections, state = [], {"chapter": "Front matter", "kind": "other", "section": None, "sub": None, "region": None,
                           "box": None}
    current = {}

    def path_of(box=None):
        parts = [state["chapter"]]
        if box:
            parts.append(box)
        else:
            parts += [x for x in (state["section"], state["sub"]) if x]
        return " > ".join(parts)

    def target(box):
        """The section that new body lines are appended to (create it on first use)."""
        key = (state["chapter"], box or state["section"], None if box else state["sub"], state["region"] if not box else None)
        if key not in current:
            kind = state["kind"]
            if box:
                kind = "box"
            elif state["section"] == "References":
                kind = "references"
            current[key] = {"chapter": state["chapter"], "region": None if box else state["region"], "box": bool(box),
                            "kind": kind, "path": path_of(box), "title": box or state["sub"] or state["section"] or state["chapter"],
                            "lines": []}
            sections.append(current[key])
        return current[key]

    for number, lines in enumerate(pages, start=1):
        pending = None                                # a heading that wraps onto a second line
        for ln in lines:
            kind = line_kind(ln["text"], ln["size"], ln["font"], ln["bold"])
            t = ln["text"]
            # --- chapter / front-matter titles (big display type)
            if ln["size"] >= 25:
                if "Cond" in ln["font"]:
                    state.update(chapter="Executive Summary" if t == "Executive Summary" else
                                 ("Contents" if "Contents" in t else "Front matter"),
                                 kind="summary" if t == "Executive Summary" else ("toc" if "Contents" in t else "other"),
                                 section=None, sub=None, region=None)
                elif re.fullmatch(r"CHAPTER \d", t):
                    state.update(chapter=CHAPTERS[t[-1]], kind="body", section=None, sub=None, region=None)
                elif t == "STATISTICAL APPENDIX":
                    state.update(chapter="Statistical Appendix", kind="other", section=None, sub=None, region=None)
                continue
            if ln["size"] >= 18 and "Helvetica" not in ln["font"]:
                continue                                # chapter title words, cover
            if kind == "section" and not ln["in_box"]:
                if pending and pending["kind"] == "section" and abs(ln["y0"] - pending["y1"]) < 6:
                    state["section"] = (state["section"] + " " + t).strip()       # heading wrapped onto two lines
                else:
                    state["section"], state["sub"] = t, None
                pending = {"kind": "section", "y1": ln["y1"]}
                state["region"] = next((r for r in REGIONS if r == state["section"]), None) or \
                    (next((r for r in REGIONS if state["section"] and r.startswith(state["section"])), None))
                continue
            if kind == "subsection" and not ln["in_box"]:
                if pending and pending["kind"] == "sub" and abs(ln["y0"] - pending["y1"]) < 6:
                    state["sub"] = (state["sub"] + " " + t).strip()
                else:
                    state["sub"] = t
                pending = {"kind": "sub", "y1": ln["y1"]}
                continue
            pending = None
            if kind == "caption":
                m = re.match(r"^BOX (\d\.\d)\s*(.*)", t)
                if m and "(continued)" not in t:
                    state["box"] = f"Box {m.group(1)} {m.group(2)}".strip()
                    target(state["box"])
                    continue
                if m:
                    continue                            # "BOX 1.1 ... (continued)" running header
                target(state["box"] if ln["in_box"] else None)["lines"].append((fixer.clean(t), number))
                continue
            if kind in ("small",):
                continue
            if ln["in_box"] and state["box"]:
                sec = target(state["box"])
            else:
                if ln["in_box"] is False and state["box"] and ln["y0"] < 80 and False:
                    pass
                sec = target(None)
            sec["lines"].append((t, number))

    # join wrapped lines into running text per section, fixing hyphens and the dropped 'Th' ligature
    out = []
    for s in sections:
        if not s["lines"]:
            continue
        pieces, buf, pg = [], [], None
        merged = []
        for text, page in s["lines"]:
            merged.append((fixer.clean(text), page))
        # join keeping the page of each line: join in page groups so citations stay accurate
        text_parts, current_page, group = [], None, []
        for text, page in merged:
            if page != current_page and group:
                text_parts.append((fixer.join(group), current_page))
                group = []
            current_page = page
            group.append(text)
        if group:
            text_parts.append((fixer.join(group), current_page))
        s["lines"] = text_parts
        out.append(s)
    return out
