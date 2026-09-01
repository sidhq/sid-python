import threading

import pytest

from sid.document_cache import DocumentCache

CONTENT = "word " * 3000  # 15,000 chars, plenty of room for disjoint seen intervals


def make_doc(tag: str):
    return {"title": f"doc {tag}", "content": CONTENT, "meta": {"tag": tag}}


def seen_view(cache: DocumentCache, data_id: str, span):
    """A view whose update_seen marks exactly `span` as seen."""
    return cache.get_single_span_document_view(data_id, "content", snippet_display_span=span)


def covers(ledger_spans, span):
    a, b = span
    return any(x <= a and b <= y for x, y in ledger_spans)


# ---------------------------------------------------------------- fork semantics


def test_fork_inherits_documents_and_seen():
    parent = DocumentCache()
    parent.add_document("doc-a", make_doc("a"))
    parent.update_seen(seen_view(parent, "doc-a", (0, 200)))

    fork, = parent.fork(1)

    assert fork.get_document("doc-a") == parent.get_document("doc-a")
    assert fork.seen_ledger["doc-a"] == [(0, 200)]


def test_fork_shares_documents_but_isolates_seen_ledgers():
    parent = DocumentCache()
    parent.add_document("doc-a", make_doc("a"))

    fork_1, fork_2 = parent.fork(2)

    fork_1.add_document("doc-b", make_doc("b"))
    fork_1.update_seen(seen_view(fork_1, "doc-a", (0, 200)))

    for other in (parent, fork_2):
        assert "doc-b" in other
        assert other.get_document("doc-b") is fork_1.get_document("doc-b")
        assert other.seen_ledger["doc-a"] == []
    assert fork_1.seen_ledger["doc-a"] == [(0, 200)]


def test_fork_documents_are_shared():
    parent = DocumentCache()
    parent.add_document("doc-a", make_doc("a"))

    fork, = parent.fork(1)
    assert fork.get_document("doc-a") is parent.get_document("doc-a")


def test_parent_usable_after_fork():
    parent = DocumentCache()
    parent.add_document("doc-a", make_doc("a"))
    children = parent.fork(2)

    parent.add_document("doc-c", make_doc("c"))
    for child in children:
        assert child.get_document("doc-c") is parent.get_document("doc-c")
    view = parent.apply_snippet("doc-a", "content", "word")
    parent.update_seen(view)
    parent._validate_mapping()
    parent._validate_store()


def test_shared_mapping_same_document_same_id():
    parent = DocumentCache()
    fork_1, fork_2 = parent.fork(2)

    id_1 = fork_1.add_document("doc-x", make_doc("x"))
    id_2 = fork_2.add_document("doc-x", make_doc("x"))
    id_p = parent.add_document("doc-x", make_doc("x"))

    assert id_1 == id_2 == id_p
    assert parent.to_data_id(id_1) == "doc-x"


def test_shared_mapping_distinct_documents_distinct_ids():
    parent = DocumentCache()
    forks = parent.fork(3)

    ids = [parent.add_document("doc-p", make_doc("p"))]
    ids += [fork.add_document(f"doc-{i}", make_doc(str(i))) for i, fork in enumerate(forks)]

    assert len(set(ids)) == len(ids)
    # ids minted in one fork resolve from every other cache
    for cache in [parent, *forks]:
        assert cache.to_data_id(ids[1]) == "doc-0"
        cache._validate_mapping()


def test_fork_of_fork():
    parent = DocumentCache()
    parent.add_document("doc-a", make_doc("a"))
    fork, = parent.fork(1)
    grandfork, = fork.fork(1)

    id_g = grandfork.add_document("doc-g", make_doc("g"))
    assert parent.to_data_id(id_g) == "doc-g"
    assert "doc-g" in parent
    assert "doc-g" in fork
    assert parent.get_document("doc-g") is grandfork.get_document("doc-g")


def test_sibling_can_read_and_snippet_document_added_after_fork():
    parent = DocumentCache()
    writer, reader = parent.fork(2)
    model_id = writer.add_document("doc-b", make_doc("b"))

    assert "doc-b" in parent
    assert "doc-b" in reader
    assert not hasattr(parent, "contains_global")
    assert reader.get_document_from_model_facing_id(model_id) is writer.get_document("doc-b")
    view = reader.apply_snippet("doc-b", "content", "word")
    assert view.data_id == "doc-b"
    assert view.snippet_display_spans


def test_seen_ledger_is_isolated_for_lazily_discovered_document():
    parent = DocumentCache()
    writer, reader = parent.fork(2)
    writer.add_document("doc-b", make_doc("b"))

    view = reader.get_single_span_document_view(
        "doc-b", "content", snippet_display_span=(0, 200)
    )
    reader.update_seen(view)

    assert reader.seen_ledger["doc-b"] == [(0, 200)]
    assert "doc-b" not in parent.seen_ledger
    assert writer.seen_ledger["doc-b"] == []


def test_add_document_deepcopies_input_exactly_once(monkeypatch):
    import sid.document_cache as document_cache_module

    original_deepcopy = document_cache_module.copy.deepcopy
    calls = []

    def recording_deepcopy(value):
        calls.append(value)
        return original_deepcopy(value)

    monkeypatch.setattr(document_cache_module.copy, "deepcopy", recording_deepcopy)
    cache = DocumentCache()
    supplied = make_doc("a")
    model_id = cache.add_document("doc-a", supplied)
    stored = cache.get_document("doc-a")

    supplied["meta"]["tag"] = "mutated by caller"
    assert stored["meta"]["tag"] == "a"
    assert calls == [supplied]

    assert cache.add_document("doc-a", make_doc("ignored")) == model_id
    cache.fork(3)
    assert calls == [supplied]


# ---------------------------------------------------------------- thread safety


def run_threads(targets):
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_concurrent_update_seen_no_lost_updates():
    n_threads, n_ops = 8, 200
    cache = DocumentCache()
    cache.add_document("doc-a", make_doc("a"))

    def worker(thread_idx):
        def run():
            for i in range(n_ops):
                start = (thread_idx * n_ops + i) * 8  # disjoint per thread and op
                cache.update_seen(seen_view(cache, "doc-a", (start, start + 5)))
        return run

    run_threads([worker(t) for t in range(n_threads)])

    ledger = cache.seen_ledger["doc-a"]
    for t in range(n_threads):
        for i in range(n_ops):
            start = (t * n_ops + i) * 8
            assert covers(ledger, (start, start + 5)), f"lost interval from thread {t}, op {i}"


def test_concurrent_adds_across_forks():
    n_threads, n_docs = 8, 200
    parent = DocumentCache()
    caches = [parent] + parent.fork(n_threads - 1)

    def worker(thread_idx, cache):
        def run():
            for i in range(n_docs):
                cache.add_document(f"doc-{thread_idx}-{i}", make_doc(f"{thread_idx}-{i}"))
        return run

    run_threads([worker(t, c) for t, c in enumerate(caches)])

    parent._validate_mapping()
    parent._validate_store()
    minted = [c.to_model_facing_id(f"doc-{t}-{i}") for t, c in enumerate(caches) for i in range(n_docs)]
    assert len(set(minted)) == n_threads * n_docs
    assert len(parent.documents) == n_threads * n_docs
    for cache in caches:
        assert cache.documents is parent.documents


def test_fork_while_writing():
    parent = DocumentCache()
    stop = threading.Event()

    def writer(tid):
        # Unique ids per thread, bounded to keep this stress test finite.
        for i in range(1000):
            if stop.is_set():
                break
            data_id = f"doc-{tid}-{i}"
            parent.add_document(data_id, make_doc(str(i)))
            parent.update_seen(seen_view(parent, data_id, (0, 100)))

    threads = [threading.Thread(target=writer, args=(tid,)) for tid in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(50):
            fork, = parent.fork(1)
            assert set(fork.seen_ledger) <= set(fork.documents)
            fork._validate_mapping()
            fork._validate_store()
    finally:
        stop.set()
        for t in threads:
            t.join()


# ---------------------------------------------------------------- snippet regression


def test_apply_snippet_short_document():
    cache = DocumentCache()
    cache.add_document("short", {"content": "just a few words"})
    view = cache.apply_snippet("short", "content", "words")
    assert view.snippet_display_spans == [(0, len("just a few words"))]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
