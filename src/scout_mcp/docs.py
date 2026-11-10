"""Search over your own notes: paragraph-bound Markdown chunks ranked with BM25.

Documents are the .md / .markdown / .txt files under the docs folder. Each document is split at blank
lines; paragraphs are merged into chunks of up to `max_chunk_chars` without ever splitting a paragraph,
and every chunk remembers the Markdown heading it sits under. The index is rebuilt automatically when a
file is added, removed or changed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from .errors import ResearchError

EXTENSIONS = {".md", ".markdown", ".txt"}
_TOKEN = re.compile(r"[^\W_]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in into is it its of on or so than that the "
    "their then there these this to was were what when where which who why will with".split()
)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


class DocsError(ResearchError):
    """A document request could not be served."""


class DocHit(BaseModel):
    doc: str
    chunk: int
    section: str
    score: float
    text: str


class DocInfo(BaseModel):
    name: str
    chars: int
    chunks: int


class DocText(BaseModel):
    name: str
    text: str
    truncated: bool
    total_chars: int


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def chunk_markdown(text: str, max_chunk_chars: int = 1200) -> list[tuple[str, str]]:
    """Split Markdown into (section heading, chunk text) pairs on paragraph boundaries."""
    chunks: list[tuple[str, str]] = []
    section = ""
    buf: list[str] = []

    def flush() -> None:
        if buf:
            chunks.append((section, "\n\n".join(buf)))
            buf.clear()

    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        block = block.strip()
        if not block:
            continue
        first, _, rest = block.partition("\n")
        heading = _HEADING.match(first)
        if heading:
            flush()
            section = heading.group(2)
            block = rest.strip()
            if not block:
                continue
        if buf and sum(len(b) for b in buf) + len(block) > max_chunk_chars:
            flush()
        buf.append(block)  # a paragraph longer than the limit stays whole
    flush()
    return chunks


@dataclass(frozen=True)
class Chunk:
    doc: str
    index: int
    section: str
    text: str


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(f"{c.section}\n{c.text}")) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / len(self.lengths) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self.tfs):
            score = 0.0
            for term in terms:
                f = tf.get(term)
                if f:
                    norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                    score += self.idf[term] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(self.chunks[i], score) for score, i in scored[:top_k]]


class DocStore:
    def __init__(self, root: Path, max_chunk_chars: int = 1200) -> None:
        self.root = Path(root).resolve()
        self.max_chunk_chars = max_chunk_chars
        self._snapshot: tuple[tuple[str, int, int], ...] | None = None
        self._index = BM25Index([])
        self._counts: dict[str, tuple[int, int]] = {}

    def _files(self) -> list[Path]:
        if not self.root.is_dir():
            return []
        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS and p.resolve().is_relative_to(self.root)
        )

    def _name(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _refresh(self) -> BM25Index:
        files = self._files()
        snapshot = tuple((self._name(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if snapshot != self._snapshot:
            chunks: list[Chunk] = []
            counts: dict[str, tuple[int, int]] = {}
            for path in files:
                name, text = self._name(path), path.read_text(encoding="utf-8", errors="replace")
                pieces = chunk_markdown(text, self.max_chunk_chars)
                chunks += [Chunk(name, i, section, body) for i, (section, body) in enumerate(pieces)]
                counts[name] = (len(text), len(pieces))
            self._index, self._counts, self._snapshot = BM25Index(chunks), counts, snapshot
        return self._index

    def search(self, query: str, top_k: int) -> list[DocHit]:
        if not tokenize(query):
            raise DocsError("The query has no searchable words; use a few specific keywords.")
        index = self._refresh()
        if not index.chunks:
            raise DocsError(f"No documents found in {self.root}. Add .md or .txt files there.")
        return [
            DocHit(doc=c.doc, chunk=c.index, section=c.section, score=round(score, 3), text=c.text)
            for c, score in index.search(query, top_k)
        ]

    def list_docs(self) -> list[DocInfo]:
        self._refresh()
        return [DocInfo(name=name, chars=chars, chunks=n) for name, (chars, n) in sorted(self._counts.items())]

    def read(self, name: str, max_chars: int) -> DocText:
        path = (self.root / name).resolve()
        if (
            not path.is_relative_to(self.root)
            or not path.is_file()
            or path.suffix.lower() not in EXTENSIONS
        ):
            raise DocsError(f"There is no document named {name!r}. Call list_docs to see the available names.")
        text = path.read_text(encoding="utf-8", errors="replace")
        return DocText(name=self._name(path), text=text[:max_chars], truncated=len(text) > max_chars, total_chars=len(text))
