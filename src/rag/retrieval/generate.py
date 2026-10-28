"""Grounded generation: numbered passages in, cited answer (or an honest refusal) out."""
from ..infra.llm import chat

REFUSAL = "I could not find this in the document."

SYSTEM_PROMPT = f"""You answer questions about one document using ONLY the numbered context passages.
- Use only facts stated in the passages. Never add outside knowledge.
- Quote numbers, names and dates exactly as written.
- Cite the passages you used with square brackets only, like [1] or [2][3]. Plain text, no bold.
- If passages give different figures, report each one with its context.
- If the passages do not contain the answer, reply exactly: {REFUSAL}
- Be brief: one to three sentences."""


def build_context(hits: list[dict]) -> str:
    """Number the passages and show where each came from, so the answer can cite them."""
    return "\n\n".join(
        f"[{i}] (pages {h['page_start']}-{h['page_end']}{', ' + h['heading_path'] if h.get('heading_path') else ''})\n{h['text']}"
        for i, h in enumerate(hits, start=1))


def ask(question: str, hits: list[dict]) -> str:
    return chat(SYSTEM_PROMPT, f"Context passages:\n\n{build_context(hits)}\n\nQuestion: {question}")
