"""Check an answer's citations against the sources the run actually read.

This is plain code, not an LLM: it finds every [S#] label, flags labels that match no source, and
measures how many claims (bullets and sentences) carry at least one label. The sources list at the
end of the answer is built here too, from the registry, so URLs can never be invented by the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sources import SourceRegistry

_GROUP = re.compile(r"\[([^\[\]]+)\]")
_LABEL = re.compile(r"\bS(\d+)\b")
_LABEL_GROUP = r"\[[^\[\]]*\bS\d+\b[^\[\]]*\]"
# "Claim one. [S1] Claim two." -> "Claim one [S1]. Claim two.": a label after the full stop belongs
# to the sentence before it, not to the next one.
_LABELS_AFTER_STOP = re.compile(rf"([.!?])\s*((?:{_LABEL_GROUP}\s*)+)")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_LIST_MARK = re.compile(r"^([-*+]|\d+[.)])\s+")


def labels_in(text: str) -> list[str]:
    """Source labels in order of first use. Accepts [S1], [S1, S3] and [S2][S4]."""
    found: list[str] = []
    for group in _GROUP.findall(text):
        for number in _LABEL.findall(group):
            label = f"S{number}"
            if label not in found:
                found.append(label)
    return found


def claims(text: str) -> list[str]:
    """Bullets and sentences of the answer that state something (headings and 'Open questions' excluded)."""
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        plain = line.lstrip("#").strip(" *_:").lower()
        if plain.startswith("open question"):
            break  # what follows are gaps, not claims
        if line.startswith("#"):
            continue
        line = _LIST_MARK.sub("", line)
        line = _LABELS_AFTER_STOP.sub(lambda m: " " + "".join(m.group(2).split()) + m.group(1) + " ", line).strip()
        out.extend(s for s in _SENTENCE_BREAK.split(line) if len(s.split()) >= 4)
    return out


@dataclass
class CitationReport:
    cited: list[str] = field(default_factory=list)  # labels that match a source the run read
    unknown: list[str] = field(default_factory=list)  # labels that match nothing: invented citations
    claims: int = 0
    cited_claims: int = 0

    @property
    def coverage(self) -> float:
        return self.cited_claims / self.claims if self.claims else 0.0


def check_citations(answer: str, registry: SourceRegistry) -> CitationReport:
    labels = labels_in(answer)
    statements = claims(answer)
    return CitationReport(
        cited=[label for label in labels if label in registry],
        unknown=[label for label in labels if label not in registry],
        claims=len(statements),
        cited_claims=sum(1 for s in statements if labels_in(s)),
    )


def render_sources(source_ids: list[str], registry: SourceRegistry) -> str:
    lines = []
    for sid in source_ids:
        source = registry.get(sid)
        if source is None:
            continue
        where = source.ref if source.kind == "web" else f"note: {source.ref}"
        lines.append(f"- [{sid}] {where}" if source.title == source.ref else f"- [{sid}] {source.title} - {where}")
    return "## Sources\n\n" + ("\n".join(lines) if lines else "No sources were cited.")
