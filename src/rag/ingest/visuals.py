"""One vision-model call per page that has images or real vector artwork (charts, icons). Cached on disk."""
import json

import pymupdf

from .. import config
from ..infra.llm import describe_image

VISION_PROMPT = """You are preparing one page of a product report for a search index.
Write plain text capturing everything a reader learns from the CHARTS, TABLES, ICONS and CALL-OUTS.
- Transcribe every number with its label, unit, and the column or group it belongs to.
- For a chart, say what each bar or segment represents and how the values relate
  (for example: baseline minus reductions equals the final value).
- Mention footnote numbers where they appear, like [note 7].
- Do not guess values that are not printed. Under 250 words."""


def vision_chunks(path, start_id: int) -> list[dict]:
    """-> chunks describing the visual pages. Pages with only plain text are skipped."""
    cache_file = config.CACHE_DIR / f"{path.stem}_vision.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    doc, out = pymupdf.open(path), []
    for number, page in enumerate(doc, start=1):
        if not page.get_images() and len(page.get_drawings()) < 5:
            continue                                                # plain text page: nothing to see
        key = str(number)
        if key not in cache:
            cache[key] = describe_image(page.get_pixmap(dpi=110).tobytes("png"), VISION_PROMPT)
            cache_file.write_text(json.dumps(cache, indent=1))
        out.append({"id": start_id + len(out), "text": cache[key], "page_start": number, "page_end": number,
                    "kind": "vision", "heading_path": f"page {number}, vision description"})
    return out
