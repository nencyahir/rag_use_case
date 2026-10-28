"""Chunking strategies. Everything that decides what a 'passage' is lives here."""
import re


def tokens(text: str) -> int:
    """Cheap token estimate: ~0.75 words per token for English prose."""
    return max(1, int(len(text.split()) / 0.75))


_ABBR = {"mr", "mrs", "ms", "dr", "st", "no", "vs", "inc", "co", "u.s", "fy", "e.g", "i.e", "etc", "al", "fig", "approx"}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) character spans of each sentence. Splits on . ! ? followed by a capital, number or bracket,
    but not after common abbreviations or single initials."""
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(\[])", text):
        last = text[start:m.start()].split(" ")[-1].rstrip(".")
        if last.lower() in _ABBR or re.fullmatch(r"[A-Z]", last):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def split_sentences(text: str) -> list[str]:
    return [text[a:b].strip() for a, b in sentence_spans(text)]


# ---------------------------------------------------------------- baseline
def naive_chunks(pages: list[dict], size: int = 1000) -> list[dict]:
    """The baseline: cut the whole text every `size` characters. Splits sentences in half."""
    text = " ".join(p["text"] for p in pages)
    chunks = []
    for i in range(0, len(text), size):
        chunks.append({"id": len(chunks), "text": text[i:i + size], "page_start": 0, "page_end": 0})
    return chunks


def naive_chunks_with_pages(pages: list[dict], size: int = 1000) -> list[dict]:
    """Same cut, but remember which page each chunk starts and ends on (needed for citations)."""
    text, marks = "", []
    for p in pages:
        marks.append((len(text), p["page"]))
        text += p["text"] + " "
    def page_at(pos):
        pg = marks[0][1]
        for off, number in marks:
            if off <= pos:
                pg = number
        return pg
    chunks = []
    for i in range(0, len(text), size):
        piece = text[i:i + size].strip()
        if piece:
            chunks.append({"id": len(chunks), "text": piece, "page_start": page_at(i), "page_end": page_at(min(i + size, len(text) - 1))})
    return chunks


# ---------------------------------------------------------------- use case 1
def sentence_chunks(pages: list[dict], target: int = 450, overlap: int = 60) -> list[dict]:
    """Whole sentences only, about `target` tokens each. The last ~`overlap` tokens repeat at the start
    of the next chunk, so a fact near a boundary appears whole somewhere."""
    sents = [(s, p["page"]) for p in pages for s in split_sentences(p["text"])]
    chunks, current, size = [], [], 0

    def close():
        nonlocal current, size
        chunks.append({"id": len(chunks), "text": " ".join(s for s, _ in current),
                       "page_start": current[0][1], "page_end": current[-1][1]})
        keep, kept = [], 0
        for s, pg in reversed(current):           # carry the tail over as overlap
            if kept + tokens(s) > overlap:
                break
            keep.insert(0, (s, pg))
            kept += tokens(s)
        current, size = keep, kept

    for s, pg in sents:
        if size + tokens(s) > target and size > overlap:
            close()
        current.append((s, pg))
        size += tokens(s)
    if size > overlap or not chunks:
        close()
    return chunks


# ---------------------------------------------------------------- use case 2
def _group(sentences, limit, overlap=0):
    """Group (sentence, page) pairs into windows of about `limit` tokens, with optional tail overlap."""
    windows, current, size, fresh = [], [], 0, 0
    for item in sentences:
        n = tokens(item[0])
        if fresh and size + n > limit:
            windows.append(current)
            keep, kept = [], 0
            for it in reversed(current):
                if kept + tokens(it[0]) > overlap:
                    break
                keep.insert(0, it)
                kept += tokens(it[0])
            current, size, fresh = keep, kept, 0
        current.append(item)
        size += n
        fresh += 1
    if fresh:
        windows.append(current)
    return windows


def section_sentences(section: dict) -> list[tuple[str, int]]:
    """Sentences of a section with the page each one starts on (a sentence may run across a page break)."""
    text, marks = "", []
    for piece, page in section["lines"]:
        marks.append((len(text), page))
        text += piece + " "
    out = []
    for a, b in sentence_spans(text):
        page = [pg for off, pg in marks if off <= a][-1]
        out.append((text[a:b].strip(), page))
    return out


def build_hierarchy(sections: list[dict], child_size: int = 280, parent_size: int = 900):
    """Parents: ~900-token windows inside one section (what the LLM will read).
    Children: ~280-token pieces of a parent (what we search). Every child knows its location."""
    children, parents = [], {}
    for s in sections:
        sentences = section_sentences(s)
        if not sentences:
            continue
        m = re.match(r"Chapter (\d)", s["chapter"])
        meta = {"chapter": s["chapter"], "chapter_no": int(m.group(1)) if m else 0,
                "region": s["region"], "box": s["box"], "kind": s["kind"], "heading_path": s["path"]}
        for window in _group(sentences, parent_size):
            pid = len(parents)
            parents[pid] = {"text": " ".join(x for x, _ in window), "page_start": window[0][1], "page_end": window[-1][1]}
            for piece in _group(window, child_size, overlap=40):
                children.append({"id": len(children), "text": " ".join(x for x, _ in piece),
                                 "page_start": piece[0][1], "page_end": piece[-1][1], "parent_id": pid, **meta})
    return children, parents


def with_heading(chunk: dict) -> str:
    """What we embed: the heading path travels with the passage (the stored text stays unchanged)."""
    return f"{chunk['heading_path']}\n{chunk['text']}"


def element_chunks(path, start_id: int = 0) -> list[dict]:
    """Brochure PDF -> one chunk per table, prose in small sentence groups, and one chunk per endnote.
    A footnote marker like [^7] in a chunk pulls the text of that endnote in with it."""
    import pymupdf
    from . import elements as el

    doc = pymupdf.open(path)
    notes = el.read_endnotes(doc)
    chunks = []

    def add(text, page, kind, heading):
        used = sorted({int(n) for n in re.findall(r"\[\^(\d+)\]", text)})
        extra = "".join(f"\n[Note {n}] {notes[n]}" for n in used if n in notes)
        chunks.append({"id": start_id + len(chunks), "text": text + extra, "page_start": page, "page_end": page,
                       "kind": kind, "heading_path": heading})

    for number, page in enumerate(doc, start=1):
        if number >= el.ENDNOTE_FIRST_PAGE:
            tables = el.rebuild_tables(page, number)
            for t in tables:
                add(t["text"], number, "table", f"page {number}, table")
            continue
        tables = el.rebuild_tables(page, number)
        for t in tables:
            add(t["text"], number, "table", f"page {number}, table")
        text = " ".join(el.prose_blocks(page, tables))
        for window in _group([(s, number) for s in split_sentences(text)], 120):
            add(" ".join(s for s, _ in window), number, "prose", f"page {number}")
    for n, body in sorted(notes.items()):
        chunks.append({"id": start_id + len(chunks), "text": f"Footnote {n}: {body}", "page_start": 11, "page_end": 12,
                       "kind": "footnote", "heading_path": f"Endnote {n}"})
    return chunks
