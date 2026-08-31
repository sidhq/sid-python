"""Invariant and format tests for the sid SDK.

Two kinds of tests live here:

* Regression tests — pin behavior the SDK gets right today: the ref grammar,
  the interval bookkeeping, the seen-masking plan, the rendered <doc> shape,
  and the IdStream guarantees. These must stay green.

* ``xfail(strict=True)`` tests — each documents a confirmed divergence from
  the trained format (as vendored in slime-tito's sam_sdk) or an API footgun.
  They fail today by design; once the underlying issue is fixed the xfail
  turns into an XPASS error, prompting removal of the marker.
"""

import random
import threading

import pytest

from sid import DocumentCache, DocumentView, DocumentViewWithSnippet, parse_rendered_model_facing_id
from sid.document_cache import insert_interval, overlap_list, plan_segments
from sid.document_rendering import render_markdown_table
from sid.id_stream import IdSpaceExhausted, IdStream

WORDS = " ".join(f"w{i:04d}" for i in range(600))  # 600 words, 6*600-1 = 3599 chars


def make_cache(content=WORDS, title="T", data_id="doc-1"):
    cache = DocumentCache()
    cache.add_document(data_id, {"title": title, "content": content})
    return cache


def read_view(cache, data_id, span):
    """A read-style view of `span` (mask=False semantics)."""
    return cache.get_single_span_document_view(
        data_id, "content", snippet_display_span=span, display_fields=["title", "content"]
    )


# =========================================================================
# The ref grammar: parse_rendered_model_facing_id
# =========================================================================


def test_parse_bare_ref():
    assert parse_rendered_model_facing_id("abcde") == ("abcde", None)


def test_parse_ranged_ref():
    assert parse_rendered_model_facing_id("abcde#800:1600") == ("abcde", (800, 1600))


@pytest.mark.parametrize("ref", ["a#", "a#x:y", "a#1:2:3", "a#1:2#3:4", "a#борис:2", "#1:2", "", None])
def test_parse_malformed_refs_raise_value_error(ref):
    # Every malformed shape is a ValueError carrying the trained message —
    # the tool layer re-raises it as the ToolError the loop feeds back
    # (see parse_reference in the reference implementation).
    with pytest.raises(ValueError):
        parse_rendered_model_facing_id(ref)


@pytest.mark.parametrize("ref", ["a#5:5", "a#10:5"])
def test_parse_rejects_empty_ranges(ref):
    with pytest.raises(ValueError, match="start must be < end"):
        parse_rendered_model_facing_id(ref)


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("a#-5:10", ("a", (-5, 10))),  # negative start clamps later, at the view
        ("a# 1:2", ("a", (1, 2))),
        ("a#+1:2", ("a", (1, 2))),
        ("a#1_0:20", ("a", (10, 20))),
    ],
)
def test_parse_offset_leniency_is_intentional(ref, expected):
    # int() accepts whitespace, signs, and underscores; being lax here means
    # fewer failed tool calls for the model. Bounds are the view's job.
    assert parse_rendered_model_facing_id(ref) == expected


# =========================================================================
# Interval bookkeeping: insert_interval / overlap_list / plan_segments
# =========================================================================


def naive_char_set(intervals):
    chars = set()
    for a, b in intervals:
        chars.update(range(a, b))
    return chars


def test_insert_interval_merges_overlap_and_touch():
    assert insert_interval([], (5, 10)) == [(5, 10)]
    assert insert_interval([(5, 10)], (8, 20)) == [(5, 20)]
    assert insert_interval([(5, 10)], (10, 20)) == [(5, 20)]  # touching merges
    assert insert_interval([(5, 10)], (20, 30)) == [(5, 10), (20, 30)]
    assert insert_interval([(5, 10), (20, 30)], (9, 21)) == [(5, 30)]


def test_insert_interval_rejects_empty():
    with pytest.raises(AssertionError):
        insert_interval([], (5, 5))


def test_insert_interval_fuzz_against_char_set_model():
    rng = random.Random(0)
    for _ in range(300):
        ledger = []
        expected = set()
        for _ in range(rng.randint(1, 12)):
            a = rng.randrange(0, 500)
            b = a + rng.randint(1, 80)
            ledger = insert_interval(ledger, (a, b))
            expected |= naive_char_set([(a, b)])
        assert naive_char_set(ledger) == expected
        # sorted, disjoint, non-touching
        for (_, y1), (x2, _) in zip(ledger, ledger[1:]):
            assert y1 < x2


def test_plan_segments_partition_invariants():
    rng = random.Random(1)
    for _ in range(500):
        ledger = []
        for _ in range(rng.randint(0, 8)):
            a = rng.randrange(0, 3000)
            ledger = insert_interval(ledger, (a, a + rng.randint(1, 400)))
        a = rng.randrange(0, 3000)
        span = (a, a + rng.randint(1, 800))
        show, seen = plan_segments(span, ledger, min_seen_overlap=100)

        # the segments exactly tile the span, in order, without overlap
        segments = sorted(show + seen, key=lambda s: s[0])
        cursor = span[0]
        for x, y in segments:
            assert x == cursor and x < y
            cursor = y
        assert cursor == span[1]

        # a masked segment is provably covered by the ledger and long enough
        for x, y in seen:
            assert overlap_list((x, y), ledger) == [(x, y)]
            assert y - x >= 100


def test_short_seen_runs_are_reshown():
    # a previously-shown stretch below min_seen_overlap re-renders as text
    cache = make_cache()
    cache.update_seen(read_view(cache, "doc-1", (0, 50)))  # 50 < 100
    view = cache.apply_snippet("doc-1", "content", "w0300", display_fields=["title", "content"])
    _, _, body = view._render_parts()
    assert "[seen:" not in body


# =========================================================================
# Snippeting: apply_snippet
# =========================================================================


def test_snippet_is_verbatim_slice():
    cache = make_cache()
    view = cache.apply_snippet("doc-1", "content", "w0300 w0301", snippet_size=20,
                               display_fields=["title", "content"])
    (a, b), = view.snippet_display_spans
    assert WORDS[a:b] == view.document["content"][a:b]
    assert "w0300" in WORDS[a:b]  # the window landed on the query terms
    assert not WORDS[a].isspace() and not WORDS[b - 1].isspace()  # word-aligned


def test_snippet_short_document_returns_whole_content():
    cache = make_cache(content="just a few words", data_id="short")
    view = cache.apply_snippet("short", "content", "words", display_fields=["title", "content"])
    assert view.snippet_display_spans == [(0, len("just a few words"))]


def test_snippet_field_must_be_displayed():
    cache = make_cache()
    with pytest.raises(ValueError, match="must be in display_fields"):
        cache.apply_snippet("doc-1", "content", "w0001", display_fields=["title"])


def test_display_fields_default_to_every_document_field():
    # The documented contract: the dict you add_document is exactly what the
    # model may see — omitting display_fields renders every field, so callers
    # must keep their database id (and anything else private) out of the dict.
    cache = DocumentCache()
    cache.add_document("d", {"title": "T", "content": "some words here", "extra": "x"})
    xml = cache.get_single_span_document_view("d", "content").render_xml()
    assert 'title="T"' in xml and 'extra="x"' in xml


# =========================================================================
# The rendered <doc> shape (the trained observation format)
# =========================================================================


def test_render_full_document_bare_ref():
    content = "aaaa bbbb cccc dddd"
    cache = make_cache(content=content)
    mid = cache.to_model_facing_id("doc-1")
    view = read_view(cache, "doc-1", None)  # None -> whole document
    assert view.render_xml() == (
        f'<doc id="{mid}" doc_length=19 title="T">\naaaa bbbb cccc dddd\n</doc>'
    )


def test_render_partial_view_ranged_ref_and_ellipses():
    content = "aaaa bbbb cccc dddd"
    cache = make_cache(content=content)
    mid = cache.to_model_facing_id("doc-1")
    view = read_view(cache, "doc-1", (5, 9))
    assert view.render_xml() == (
        f'<doc id="{mid}#5:9" doc_length=19 title="T">\n... bbbb ...\n</doc>'
    )


def test_render_masks_seen_stretch_with_marker():
    cache = make_cache()
    cache.update_seen(read_view(cache, "doc-1", (0, 150)))
    mid = cache.to_model_facing_id("doc-1")
    view = cache.apply_snippet("doc-1", "content", "w0550", snippet_size=700,
                               display_fields=["title", "content"])
    # 600 words <= 700 -> the snippet is the whole document, bare ref
    doc_id, attrs, body = view._render_parts()
    assert doc_id == mid
    assert body == '[seen: "#0:150"]' + WORDS[150:]
    assert attrs["doc_length"] == len(WORDS)


def test_pure_repeat_compacts_to_single_marker():
    cache = make_cache(content="just a few words", data_id="short")
    first = cache.apply_snippet("short", "content", "words", display_fields=["title", "content"])
    cache.update_seen(first)
    repeat = cache.apply_snippet("short", "content", "words", display_fields=["title", "content"])
    assert repeat.snippet_display_spans == []
    doc_id, _, body = repeat._render_parts()
    assert doc_id == cache.to_model_facing_id("short")  # bare ref, never ranged
    assert body == '[seen: "#0:16"]'


def test_render_attr_conventions():
    # ints unquoted, lists comma-joined, falsy values skipped, attrs escaped
    cache = DocumentCache()
    cache.add_document("d", {"title": 'A "quoted" <title>', "tags": ["x", "y"], "count": 3,
                             "empty": "", "content": "words here"})
    mid = cache.to_model_facing_id("d")
    view = cache.get_single_span_document_view(
        "d", display_fields=["title", "tags", "count", "empty"]
    )
    assert view.render_xml() == (
        f'<doc id="{mid}" title="A "quoted" &lt;title&gt;" tags="x, y" count=3></doc>'
    )


def test_update_seen_accumulates_and_merges():
    cache = make_cache()
    cache.update_seen(read_view(cache, "doc-1", (0, 150)))
    cache.update_seen(read_view(cache, "doc-1", (100, 300)))
    cache.update_seen(read_view(cache, "doc-1", (500, 600)))
    assert cache.seen_ledger["doc-1"] == [(0, 300), (500, 600)]


# =========================================================================
# IdStream
# =========================================================================


def test_id_stream_covers_space_exactly_then_raises():
    stream = IdStream(alphabet="ab", length=3, seed=0)
    ids = [next(stream) for _ in range(8)]
    assert len(set(ids)) == 8
    assert set(ids) == {a + b + c for a in "ab" for b in "ab" for c in "ab"}
    with pytest.raises(IdSpaceExhausted):
        next(stream)


def test_id_stream_getitem_is_bijective_and_stable():
    stream = IdStream(alphabet="abcd", length=2, seed=7)
    all_ids = [stream[i] for i in range(16)]
    assert len(set(all_ids)) == 16
    assert stream.minted == 0  # __getitem__ consumes nothing
    assert [next(stream) for _ in range(16)] == all_ids


def test_id_stream_seed_reproducible():
    a = IdStream(seed=42)
    b = IdStream(seed=42)
    assert [next(a) for _ in range(50)] == [next(b) for _ in range(50)]


def test_id_stream_rejects_separator_alphabet():
    with pytest.raises(ValueError):
        IdStream(alphabet="abc#")
    with pytest.raises(ValueError):
        IdStream(alphabet="abc:")
    with pytest.raises(ValueError):
        IdStream(alphabet="abca")  # duplicate chars


def test_id_stream_thread_safe():
    stream = IdStream(seed=0)
    minted = []
    lock = threading.Lock()

    def worker():
        local = [next(stream) for _ in range(500)]
        with lock:
            minted.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(minted)) == 8 * 500


# =========================================================================
# Markdown table rendering
# =========================================================================


def test_markdown_table_shape_and_escaping():
    cache = DocumentCache()
    cache.add_document("d1", {"title": "pipe | inside", "content": "line one\nline two"})
    mid = cache.to_model_facing_id("d1")
    view = cache.get_single_span_document_view(
        "d1", "content", display_fields=["title", "content"]
    )
    table = render_markdown_table([view])
    lines = table.splitlines()
    assert lines[0] == "| id | doc_length | title | content |"
    assert lines[2] == f"| {mid} | 17 | pipe \\| inside | line one line two |"


def test_markdown_table_empty():
    assert render_markdown_table([]) == ""


# =========================================================================
# API contract: loud failures and clamping
# =========================================================================


def test_add_document_returns_model_facing_id():
    cache = DocumentCache()
    mid = cache.add_document("d", {"content": "x y z"})
    assert isinstance(mid, str)
    assert cache.to_data_id(mid) == "d"


def test_re_adding_a_document_fails_loudly():
    # The seen ledger's offsets index the first content string; a silent
    # replace would reset what the model has already been shown.
    cache = make_cache()
    cache.update_seen(read_view(cache, "doc-1", (0, 200)))
    with pytest.raises(ValueError, match="add-once"):
        cache.add_document("doc-1", {"title": "T", "content": WORDS})
    assert cache.seen_ledger["doc-1"] == [(0, 200)]


def test_empty_content_raises_clear_error():
    cache = DocumentCache()
    cache.add_document("d", {"content": ""})
    with pytest.raises(TypeError, match="non-empty string"):
        cache.apply_snippet("d", "content", "q", display_fields=["content"])
    with pytest.raises(TypeError, match="non-empty string"):
        cache.get_single_span_document_view("d", "content", display_fields=["content"])


def test_overlong_span_is_clamped():
    # The model may ask for 0:20000 of a shorter document: clamp and return
    # correctly — the ranged ref names the clamped (verbatim) span.
    cache = DocumentCache()
    cache.add_document("d", {"content": "short text here"})
    mid = cache.to_model_facing_id("d")
    view = cache.get_single_span_document_view(
        "d", "content", snippet_display_span=(5, 5000), display_fields=["content"]
    )
    assert view.snippet_display_spans == [(5, 15)]
    assert view.render_xml() == f'<doc id="{mid}#5:15" doc_length=15>\n...  text here\n</doc>'
    cache.update_seen(view)
    assert cache.seen_ledger["d"] == [(5, 15)]


def test_negative_start_is_clamped():
    cache = DocumentCache()
    cache.add_document("d", {"content": "short text here"})
    view = cache.get_single_span_document_view(
        "d", "content", snippet_display_span=(-5, 10), display_fields=["content"]
    )
    assert view.snippet_display_spans == [(0, 10)]


def test_wholly_out_of_bounds_span_is_an_integration_error():
    # No intersection with the document -> nothing to clamp to. The cache only
    # sees pre-validated spans: bounds-checking model input (with the trained
    # out-of-bounds message) is the tool layer's job, so this is a ValueError,
    # not a ToolError.
    cache = DocumentCache()
    cache.add_document("d", {"content": "short text here"})
    with pytest.raises(ValueError, match="entirely outside"):
        cache.get_single_span_document_view(
            "d", "content", snippet_display_span=(800, 1600), display_fields=["content"]
        )


def test_contains_model_facing_id():
    cache = DocumentCache()
    assert not cache.contains_model_facing_id("zzzzz")

    mid = cache.add_document("data-1", {"content": "some text"})
    assert cache.contains_model_facing_id(mid)

    # the id mapping is shared across the fork family
    fork, = cache.fork(1)
    assert fork.contains_model_facing_id(mid)
    fork_mid = fork.add_document("data-2", {"content": "more text"})
    assert cache.contains_model_facing_id(fork_mid)


def test_snap_char_range():
    cache = DocumentCache()
    content = "alpha bravo charlie delta"
    cache.add_document("d", {"content": content})

    # overlong end clamps to the document, then snaps to word boundaries
    assert cache.snap_char_range("d", "content", (6, 20000)) == (6, len(content))
    # negative start clamps to 0
    assert cache.snap_char_range("d", "content", (-5, 11)) == (0, 11)
    # mid-word edges snap outward
    assert cache.snap_char_range("d", "content", (8, 14)) == (6, 19)
    # a range entirely outside the document falls back to the full document
    # (deliberate training-inference divergence), on either side
    assert cache.snap_char_range("d", "content", (800, 1600)) == (0, len(content))
    assert cache.snap_char_range("d", "content", (-10, -5)) == (0, len(content))
    # unknown ids and fields are integration errors, not model mistakes
    with pytest.raises(KeyError):
        cache.snap_char_range("ghost", "content", (0, 5))
    with pytest.raises(KeyError):
        cache.snap_char_range("d", "no_such_field", (0, 5))


def test_inverted_span_rejected():
    cache = DocumentCache()
    cache.add_document("d", {"content": "short text here"})
    with pytest.raises(ValueError, match="start must be < end"):
        cache.get_single_span_document_view(
            "d", "content", snippet_display_span=(10, 5), display_fields=["content"]
        )


def test_update_seen_on_snippetless_view_is_a_noop():
    # A snippet-less view displays attribute fields only — no content span was
    # shown, so there is nothing to record and the ledger must not lie.
    cache = DocumentCache()
    cache.add_document("d", {"title": "t", "content": "some text"})
    view = cache.get_single_span_document_view("d", display_fields=["title"])
    cache.update_seen(view)
    assert cache.seen_ledger["d"] == []


# =========================================================================
# Known divergences from the trained format, accepted for now.
# Each is xfail(strict=True): it fails today, and starts erroring (XPASS)
# the moment the underlying issue is fixed — then delete the marker.
# =========================================================================

format_drift = pytest.mark.xfail(
    strict=True, reason="diverges from the trained observation format (sam_sdk vendored code)"
)


@format_drift
def test_body_is_html_escaped():
    # sam_sdk escaped &, <, > in the body (html.escape(quote=False)); the
    # model was trained on escaped bodies. render_xml emits the body raw.
    content = "a <b>bold</b> & escaped body " + "filler " * 30
    cache = make_cache(content=content)
    view = read_view(cache, "doc-1", None)
    body = view.render_xml().split("\n", 1)[1]
    assert "&lt;b&gt;" in body and "&amp;" in body


@format_drift
def test_fully_seen_row_drops_display_attrs():
    # sam_sdk rendered a pure repeat as `<doc id=... doc_length=N>` with no
    # other attributes; render_xml keeps title etc.
    cache = make_cache(content="just a few words", data_id="short")
    first = cache.apply_snippet("short", "content", "words", display_fields=["title", "content"])
    cache.update_seen(first)
    repeat = cache.apply_snippet("short", "content", "words", display_fields=["title", "content"])
    assert "title=" not in repeat.render_xml()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
