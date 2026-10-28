"""Query understanding: turn place names in a question into metadata filters (rules, or one LLM call)."""
import json
import re

from qdrant_client import models

from ..config import REGIONS
from ..infra.llm import chat

NOISE = ["header", "references", "toc"]          # chunk kinds we never want to retrieve


def to_filter(region=None, chapter=None, executive_summary=False, box=False):
    """Metadata -> Qdrant filter. Every chunk was stored with chapter, region and kind."""
    must = []
    if region:
        must.append(models.FieldCondition(key="region", match=models.MatchValue(value=region)))
    if executive_summary:
        must.append(models.FieldCondition(key="chapter", match=models.MatchValue(value="Executive Summary")))
    elif chapter:
        must.append(models.FieldCondition(key="chapter_no", match=models.MatchValue(value=int(chapter))))
    if box:
        must.append(models.FieldCondition(key="kind", match=models.MatchValue(value="box")))
    skip = [models.FieldCondition(key="kind", match=models.MatchAny(any=NOISE))]
    return models.Filter(must=must or None, must_not=skip)


def route_rules(question: str) -> dict:
    """Regular expressions: cheap and predictable, but they only know the exact names we listed."""
    tags = {}

    def mentions(alias):
        if len(alias) <= 6 and " " not in alias and alias != "mena":    # acronyms must be upper-case: "SAR", "EAP"
            return re.search(rf"\b{alias.upper()}\b", question)
        return re.search(rf"\b{re.escape(alias)}\b", question, re.IGNORECASE)

    regions = [r for r, aliases in REGIONS.items() if any(mentions(a) for a in aliases)]
    if len(regions) == 1:
        tags["region"] = regions[0]
    if re.search(r"\bexecutive summary\b", question, re.IGNORECASE):
        tags["executive_summary"] = True
    elif m := re.search(r"\bchapter\s+(\d)\b", question, re.IGNORECASE):
        tags["chapter"] = int(m.group(1))
    if re.search(r"\bbox\s+\d\.\d\b", question, re.IGNORECASE):
        tags["box"] = True
    return tags


QUERY_PROMPT = f"""You turn a question about the World Bank's Global Economic Prospects report (June 2026)
into search instructions. Every passage in the report is tagged with where it lives:

- region (Chapter 2 has one section per region): {", ".join(REGIONS)}
- chapter: 1 Global Outlook, 2 Regional Outlooks, 3 Sovereign Debt Levels and Interest Rates in EMDEs,
  4 Fiscal Policy and Commodity Price Swings
- the Executive Summary, and numbered boxes such as "Box 1.1"

Reply with JSON only:
{{"region": "<one region exactly as listed, or null>",
  "chapter": <1, 2, 3, 4 or null>,
  "executive_summary": <true or false>,
  "box": <true if the question names a box, else false>,
  "search_query": "<the question rewritten for search>"}}

Rules:
- Set region when the question names a region OR a country in it (India -> South Asia, Brazil -> Latin America and the Caribbean,
  Pakistan -> Middle East, North Africa, Afghanistan, and Pakistan).
- Global or multi-region questions get region null.
- Set chapter only when the question literally says "Chapter N". Never infer it from the topic:
  a wrong filter hides the answer, a missing one only makes search a little slower to find it.
- search_query: spell out acronyms and use the report's words (EMDEs, outlook, projected growth).
  Keep every number and year. Do not add the report's title."""


def understand(question: str):
    """One LLM call: question -> filter fields + a search-friendly rewrite. Unknown values are dropped."""
    reply = chat(QUERY_PROMPT, question)
    found = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    try:
        data = json.loads(found.group(0)) if found else {}
    except json.JSONDecodeError:
        data = {}
    tags = {}
    if data.get("region") in REGIONS:                                   # never trust a name we did not list
        tags["region"] = data["region"]
    if data.get("executive_summary") is True:
        tags["executive_summary"] = True
    elif data.get("chapter") in (1, 2, 3, 4):
        tags["chapter"] = data["chapter"]
    if data.get("box") is True:
        tags["box"] = True
    return tags, (data.get("search_query") or question).strip()


def plan(question: str, use_llm: bool = True):
    """-> (qdrant filter, tags, search query, 'llm' | 'rules'). Falls back to rules if the LLM call fails."""
    if use_llm:
        try:
            tags, search_query = understand(question)
            return to_filter(**tags), tags, search_query, "llm"
        except Exception:
            pass
    tags = route_rules(question)
    return to_filter(**tags), tags, question, "rules"
