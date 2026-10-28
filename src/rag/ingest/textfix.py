"""Repairs for text pulled out of PDFs: ligature glyphs, dropped 'Th' ligatures, line-break hyphens."""
import re
from collections import Counter

LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st"})
_WORD = re.compile(r"\b[a-z][A-Za-z]*")       # lower-case-initial tokens only: a dropped-ligature 'Te' must not count as a word


class Fixer:
    """Learns the document's own vocabulary, then uses it to decide ambiguous repairs."""

    def __init__(self, raw_text: str):
        text = raw_text.translate(LIGATURES)
        self.vocab = Counter(w.lower() for w in _WORD.findall(text))

    def th_ligature(self, text: str) -> str:
        """Some fonts drop the 'Th' ligature glyph, leaving 'Te conflict', 'Tis', 'Tese'.
        Restore 'T'+x -> 'Th'+x only when 'th'+x is a word this document uses and 't'+x is not."""
        def fix(m):
            rest = m.group(1)
            if self.vocab.get("th" + rest.lower(), 0) > 0 and self.vocab.get("t" + rest.lower(), 0) == 0:
                return "Th" + rest
            return m.group(0)
        return re.sub(r"\bT([a-z]{1,8})\b", fix, text)

    def join(self, lines: list[str]) -> str:
        """Join wrapped lines into running text, repairing 'sustainabil-' + 'ity' style breaks."""
        out = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not out:
                out = line
            elif out.endswith("-") and line[:1].islower():
                head = out[:-1].split(" ")[-1].split("-")[-1]
                tail = re.match(r"[A-Za-z]+", line)
                glued = (head + (tail.group(0) if tail else "")).lower()
                if self.vocab.get(glued, 0) > 0 or self.vocab.get(head.lower(), 0) == 0:
                    out = out[:-1] + line                  # a real word split by the line break
                else:
                    out = out + line                       # a genuine hyphenated compound
            elif out.endswith("-") and line[:1].isupper():
                out = out + line                           # "Vance-" + "Cooks"
            else:
                out = out + " " + line
        return out

    def clean(self, text: str, th: bool = True) -> str:
        text = text.translate(LIGATURES)
        text = re.sub(r"[ \t ]+", " ", text)
        return self.th_ligature(text) if th else text
