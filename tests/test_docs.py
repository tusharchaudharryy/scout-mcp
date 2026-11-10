import os
import time

import pytest

from scout_mcp.docs import BM25Index, Chunk, DocsError, DocStore, chunk_markdown, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Red-Palm weevil, in 2025!") == ["red", "palm", "weevil", "2025"]


def test_chunks_follow_paragraphs_and_keep_their_heading():
    text = "# Title\n\nFirst para.\n\nSecond para.\n\n## Part two\n\nThird para."
    assert chunk_markdown(text, max_chunk_chars=1000) == [
        ("Title", "First para.\n\nSecond para."),
        ("Part two", "Third para."),
    ]


def test_long_paragraph_is_never_split():
    long = "word " * 400
    chunks = chunk_markdown(f"short\n\n{long}\n\nshort again", max_chunk_chars=100)
    assert [len(body) for _, body in chunks][1] == len(long.strip())
    assert len(chunks) == 3


def test_bm25_prefers_the_chunk_that_matches_more_rare_terms():
    index = BM25Index([
        Chunk("a.md", 0, "", "palm trees need water"),
        Chunk("b.md", 0, "", "acoustic sensors detect weevil larvae in palm trunks"),
        Chunk("c.md", 0, "", "tractor maintenance schedule"),
    ])
    results = index.search("acoustic weevil detection", top_k=3)
    assert [c.doc for c, _ in results] == ["b.md"]


def test_search_returns_section_and_skips_non_text_files(docs_dir):
    store = DocStore(docs_dir)
    hits = store.search("acoustic larvae", top_k=3)
    assert hits[0].doc == "pests.md"
    assert hits[0].section == "Detection"
    assert all(not h.doc.endswith(".png") for h in hits)


def test_query_without_searchable_words_is_rejected(docs_dir):
    with pytest.raises(DocsError, match="no searchable words"):
        DocStore(docs_dir).search("the of and", top_k=3)


def test_empty_folder_gives_a_helpful_error(tmp_path):
    with pytest.raises(DocsError, match="No documents found"):
        DocStore(tmp_path / "missing").search("palm", top_k=3)


@pytest.mark.parametrize("name", ["../secret.txt", "image.png", "nope.md", str(os.path.abspath(os.sep))])
def test_read_rejects_unknown_or_outside_files(docs_dir, name):
    with pytest.raises(DocsError, match="no document named"):
        DocStore(docs_dir).read(name, max_chars=1000)


def test_read_truncates_and_reports_length(docs_dir):
    doc = DocStore(docs_dir).read("pests.md", max_chars=20)
    assert doc.truncated and len(doc.text) == 20 and doc.total_chars > 20


def test_index_picks_up_new_and_changed_files(docs_dir):
    store = DocStore(docs_dir)
    assert not any(h.doc == "new.md" for h in store.search("greenhouse", top_k=5) or [])  # nothing yet
    (docs_dir / "new.md").write_text("Greenhouse humidity notes.", encoding="utf-8")
    assert store.search("greenhouse", top_k=5)[0].doc == "new.md"
    time.sleep(0.01)
    (docs_dir / "new.md").write_text("Now about hydroponics only.", encoding="utf-8")
    assert store.search("greenhouse", top_k=5) == []
    assert [d.name for d in store.list_docs()] == ["irrigation.md", "misc.txt", "new.md", "pests.md"]
