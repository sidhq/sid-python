import concurrent.futures
import json
from pathlib import Path

import pytest
import sid._snippet as native_snippet

from sid.document_cache import (
    SUPPORTED_LANGUAGES,
    DocumentCache,
    bm25_snippet_with_stride,
)


def words(count: int) -> str:
    return " ".join(f"w{i:04d}" for i in range(count))


def test_direct_helper_returns_exactly_fifty_source_tokens():
    content = words(100)
    start, end = bm25_snippet_with_stride("w0075", content)
    selected = content[start:end]
    assert len(selected.split()) == 50
    assert "w0075" in selected
    assert not selected[0].isspace() and not selected[-1].isspace()


def test_native_module_ships_typing_markers():
    package = Path(native_snippet.__file__).parent
    assert (package / "_snippet.pyi").is_file()
    assert (package / "py.typed").is_file()


def test_phrase_evidence_beats_scattered_terms():
    content = (
        "contingent filler filler fee filler filler agreement pad pad pad "
        "contingent fee agreement tail tail tail"
    )
    start, end = bm25_snippet_with_stride(
        "contingent fee agreement", content, window_size=6, stride=2
    )
    assert "contingent fee agreement" in content[start:end]


def test_relevance_migration_fixtures_choose_expected_regions():
    fixture_path = Path(__file__).parent / "fixtures/snippet_relevance.json"
    fixtures = json.loads(fixture_path.read_text())
    for case in fixtures:
        start, end = bm25_snippet_with_stride(
            case["query"],
            case["content"],
            window_size=case["window_size"],
            stride=case["stride"],
        )
        assert case["expected_text"] in case["content"][start:end], case["name"]



def test_cache_language_default_override_and_generic():
    # German removes "und", so the first alpha/beta pair ties on the bridged
    # bigram and wins earliest. Generic keeps it, so only the later phrase has
    # both query bigrams.
    content = "alpha beta und x x alpha und beta x x x x"
    cache = DocumentCache(language="german")
    cache.add_document("d", {"content": content})

    inherited = cache.apply_snippet("d", "content", "alpha und beta", snippet_size=4)
    overridden = cache.apply_snippet(
        "d", "content", "alpha und beta", snippet_size=4, language="generic"
    )

    assert cache.language == "german"
    assert inherited.snippet_display_spans != overridden.snippet_display_spans


def test_fork_inherits_language():
    parent = DocumentCache(language="swedish")
    child, = parent.fork(1)
    assert child.language == "swedish"


def test_every_supported_language_and_invalid_language():
    for language in SUPPORTED_LANGUAGES:
        assert DocumentCache(language=language).language == language
    with pytest.raises(ValueError, match="supported values:.*english.*generic"):
        DocumentCache(language="klingon")
    with pytest.raises(ValueError, match="supported values"):
        bm25_snippet_with_stride("q", "some content", language="klingon")


@pytest.mark.parametrize(("window_size", "stride"), [(0, 1), (-1, 1), (2, 0), (2, -1)])
def test_direct_helper_rejects_nonpositive_sizes(window_size, stride):
    with pytest.raises(ValueError):
        bm25_snippet_with_stride(
            "query", "one two three", window_size=window_size, stride=stride
        )


@pytest.mark.parametrize(
    "content",
    [
        "😀 zero one café target three four",
        "e\u0301 zero one target three four",
        "漢字 zero one target three four",
        "ภาษาไทย zero one target three four",
    ],
)
def test_unicode_offsets_slice_original_python_string(content):
    start, end = bm25_snippet_with_stride(
        "target", content, window_size=3, stride=1
    )
    assert "target" in content[start:end]
    assert 0 <= start < end <= len(content)


def test_rendering_and_seen_masking_remain_compatible():
    content = words(100)
    cache = DocumentCache()
    cache.add_document("d", {"title": "T", "content": content})
    first = cache.apply_snippet("d", "content", "w0075", snippet_size=20)
    cache.update_seen(first)
    repeat = cache.apply_snippet("d", "content", "w0075", snippet_size=20)
    assert repeat.snippet_display_spans == []
    assert "[seen:" in repeat.render_xml()


def test_concurrent_native_calls_are_deterministic():
    content = words(500)

    def call():
        return bm25_snippet_with_stride("w0400 w0401", content, 50, 10)

    expected = call()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        assert set(executor.map(lambda _: call(), range(200))) == {expected}
