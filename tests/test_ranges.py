import random
from typing import get_args

import pytest

from sid import DocumentCache, InvalidCharacterRange, RangeMode


CONTENT = "alpha bravo charlie delta"


def make_cache(*, range_mode="lenient"):
    cache = DocumentCache(range_mode=range_mode)
    cache.add_document("d", {"title": "T", "content": CONTENT})
    return cache


def test_range_mode_is_public_typed_configuration():
    assert set(get_args(RangeMode)) == {"lenient", "strict"}
    assert DocumentCache().range_mode == "lenient"
    assert DocumentCache(range_mode="strict").range_mode == "strict"
    with pytest.raises(ValueError, match="supported values: 'lenient', 'strict'"):
        DocumentCache(range_mode="permissive")


@pytest.mark.parametrize("range_mode", ["lenient", "strict"])
def test_fork_inherits_range_mode(range_mode):
    cache = DocumentCache(language="swedish", range_mode=range_mode)
    child, = cache.fork(1)
    grandchild, = child.fork(1)
    assert child.range_mode == range_mode
    assert grandchild.range_mode == range_mode
    assert child.language == "swedish"


@pytest.mark.parametrize("char_range", [(0, 1), (0, len(CONTENT)), (8, 14)])
@pytest.mark.parametrize("range_mode", ["lenient", "strict"])
def test_valid_range_is_returned_exactly(range_mode, char_range):
    cache = make_cache(range_mode=range_mode)
    assert cache.resolve_char_range("d", "content", char_range) == char_range


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ((-5, 10), (0, 10)),
        ((5, 999), (5, len(CONTENT))),
        ((-5, 999), (0, len(CONTENT))),
    ],
)
def test_default_range_handling_returns_exact_overlap(requested, expected):
    cache = make_cache()
    assert cache.resolve_char_range("d", "content", requested) == expected


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        ((-5, 10), "start -5 must be at least 0"),
        ((5, 999), "end 999 exceeds the document length"),
    ],
)
def test_exact_range_validation_rejects_out_of_bounds(requested, message):
    cache = make_cache(range_mode="strict")
    with pytest.raises(InvalidCharacterRange, match=message):
        cache.resolve_char_range("d", "content", requested)


def test_single_span_view_uses_configured_exact_validation():
    cache = make_cache(range_mode="strict")
    with pytest.raises(InvalidCharacterRange, match="start -5 must be at least 0"):
        cache.get_single_span_document_view(
            "d",
            "content",
            snippet_display_span=(-5, 10),
            display_fields=["content"],
        )


@pytest.mark.parametrize("range_mode", ["lenient", "strict"])
@pytest.mark.parametrize("requested", [(100, 200), (-20, -1)])
def test_disjoint_ranges_are_always_rejected(range_mode, requested):
    cache = make_cache(range_mode=range_mode)
    with pytest.raises(InvalidCharacterRange, match="does not overlap the document"):
        cache.resolve_char_range("d", "content", requested)


@pytest.mark.parametrize("requested", [(5, 5), (10, 5)])
@pytest.mark.parametrize("range_mode", ["lenient", "strict"])
def test_empty_and_inverted_ranges_are_always_rejected(range_mode, requested):
    cache = make_cache(range_mode=range_mode)
    with pytest.raises(InvalidCharacterRange, match=r"start \d+ must be less than end \d+"):
        cache.resolve_char_range("d", "content", requested)


@pytest.mark.parametrize(
    "requested",
    [
        None,
        (0,),
        (0, 1, 2),
        [0, 1],
        (False, 1),
        (0, True),
        ("0", 1),
        (0, 1.5),
    ],
)
def test_malformed_ranges_raise_the_public_error(requested):
    cache = make_cache()
    with pytest.raises(InvalidCharacterRange):
        cache.resolve_char_range("d", "content", requested)


def test_range_errors_are_clear_without_policy_or_repair_details():
    cases = [
        (make_cache(range_mode="strict"), (-5, 10)),
        (make_cache(range_mode="strict"), (5, 999)),
        (make_cache(), (100, 200)),
        (make_cache(), (10, 5)),
        (make_cache(), (False, 5)),
    ]
    forbidden = ("strict", "lenient", "clamp", "intersect", "adjust")

    for cache, requested in cases:
        with pytest.raises(InvalidCharacterRange) as raised:
            cache.resolve_char_range("d", "content", requested)
        message = str(raised.value).lower()
        assert "invalid character range" in message
        assert "document 'd'" in message
        assert not any(word in message for word in forbidden)


def test_midword_view_uses_exact_slice_in_id_body_and_seen_ledger():
    cache = make_cache()
    model_id = cache.to_model_facing_id("d")
    view = cache.get_single_span_document_view(
        "d",
        "content",
        snippet_display_span=(8, 14),
        display_fields=["title", "content"],
    )

    assert CONTENT[8:14] == "avo ch"
    assert view.snippet_display_spans == [(8, 14)]
    assert view.render_xml() == (
        f'<doc id="{model_id}#8:14" doc_length=25 title="T">\n'
        "... avo ch ...\n"
        "</doc>"
    )
    cache.update_seen(view)
    assert cache.seen_ledger["d"] == [(8, 14)]


@pytest.mark.parametrize("range_mode", ["lenient", "strict"])
def test_none_span_still_means_the_complete_nonempty_field(range_mode):
    cache = make_cache(range_mode=range_mode)
    view = cache.get_single_span_document_view(
        "d", "content", snippet_display_span=None, display_fields=["content"]
    )
    assert view.snippet_display_spans == [(0, len(CONTENT))]


def test_default_range_resolution_randomized_slice_invariants():
    rng = random.Random(8128)
    content = "é🙂漢字 combining e\N{COMBINING ACUTE ACCENT} " * 4
    cache = DocumentCache()
    cache.add_document("unicode", {"content": content})

    for _ in range(500):
        start = rng.randint(-2 * len(content), len(content) - 1)
        end = rng.randint(max(start + 1, 1), 3 * len(content))
        resolved = cache.resolve_char_range("unicode", "content", (start, end))
        expected = (max(0, start), min(end, len(content)))
        assert resolved == expected
        a, b = resolved
        assert 0 <= a < b <= len(content)
        assert content[a:b] == content[resolved[0]:resolved[1]]


class InvalidSelectorCache(DocumentCache):
    def __init__(self, selected_span):
        super().__init__(range_mode="lenient")
        self.selected_span = selected_span

    def snippet_fn(self, query, content, snippet_size, language="english"):
        return self.selected_span


@pytest.mark.parametrize("selected_span", [(-1, 5), (0, 999), (5, 5), (999, 1000)])
def test_selector_spans_are_validated_exactly(selected_span):
    cache = InvalidSelectorCache(selected_span)
    cache.add_document("d", {"content": CONTENT})
    with pytest.raises(InvalidCharacterRange):
        cache.apply_snippet("d", "content", "query", display_fields=["content"])
