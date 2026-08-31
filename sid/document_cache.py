import copy
from threading import Lock
from typing import List, Dict, Tuple, Any

from .document_rendering import DocumentView, DocumentViewWithSnippet
from .id_stream import IdStream
from ._snippet import bm25_snippet_with_stride

# Snippet size in Alyze UAX #29 source-token positions.
SNIPPET_SIZE_DEFAULT = 50
MIN_SEEN_OVERLAP_DEFAULT = 100

Interval = Tuple[int, int]

SUPPORTED_LANGUAGES = (
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "generic",
    "hungarian",
    "italian",
    "norwegian",
    "portuguese",
    "russian",
    "spanish",
    "swedish",
)


def stride_size(window_size: int) -> int:
    """The source-token stride used when sliding a snippet window:
    a fifth of the window, and at least one token."""
    return max(1, window_size // 5)


def _validate_language(language: str) -> str:
    if language not in SUPPORTED_LANGUAGES:
        supported = ", ".join(SUPPORTED_LANGUAGES)
        raise ValueError(
            f"unsupported language {language!r}; supported values: {supported}"
        )
    return language


def insert_interval(intervals: List[Interval], new: Interval) -> List[Interval]:
    """Insert ``new`` into a list of disjoint intervals, merging every
    interval it touches, and return the result sorted and disjoint again."""
    a, b = new
    assert a < b, f"cannot record empty interval {new}"
    out: List[Interval] = []
    placed = False
    for x, y in intervals:
        if y < a or x > b:
            if x > b and not placed:
                out.append((a, b))
                placed = True
            out.append((x, y))
        else:
            a, b = min(a, x), max(b, y)
    if not placed:
        out.append((a, b))
    return sorted(out)


def overlap(span: Interval, interval: Interval) -> Interval:
    """The intersection of ``span`` and ``interval``, or ``None`` if they
    do not overlap."""
    a, b = span
    x, y = interval
    lo, hi = max(a, x), min(b, y)
    if lo < hi:
        return (int(lo), int(hi))
    return None


def overlap_list(span: Interval, intervals: List[Interval]) -> List[Interval]:
    """The parts of ``span`` covered by ``intervals``: every non-empty
    intersection of ``span`` with an interval in the list, in order."""
    return [
        overlap(span, interval) for interval in intervals if overlap(span, interval) is not None
    ]


def append_coalescing(spans: List[Interval], new: Interval):
    """Append ``new`` to ``spans`` in place, merging it into the last span
    when the two are adjacent."""
    x, y = new
    if spans:
        a, b = spans[-1]
        if b == x:
            spans[-1] = (a, y)
            return
    spans.append((x, y))


def plan_segments(
    span: Interval, seen: List[Interval], min_seen_overlap: int
) -> Tuple[List[Interval], List[Interval]]:
    """Split ``span`` into what to display and what to mask, given the
    ``seen`` intervals already shown to the model.

    Returns ``(display_spans, seen_spans)``, together covering ``span`` in
    order: seen stretches of at least ``min_seen_overlap`` characters are
    masked (they render as ``[seen: ...]`` markers), shorter ones are simply
    displayed again rather than paying a marker for a few characters."""
    a, b = span
    display_spans: List[Interval] = []
    seen_spans: List[Interval] = []
    cursor = a
    for x, y in overlap_list(span, seen):
        if x > cursor:
            append_coalescing(display_spans, (cursor, x))
        if y - x >= min_seen_overlap:
            seen_spans.append((x, y))
        else:
            append_coalescing(display_spans, (x, y))
        cursor = y
    if cursor < b:
        append_coalescing(display_spans, (cursor, b))
    return display_spans, seen_spans


class DocumentCache:
    def __init__(self, language: str = "english"):
        """Create a cache whose snippets use ``language`` by default.

        Named languages lowercase tokens and apply Alyze's language-specific
        stopword list. ``generic`` lowercases without removing stopwords.
        Individual ``apply_snippet`` calls may override this setting.
        """
        self.language = _validate_language(language)
        # Shared by reference across forks, guarded by the shared _mapping_lock.
        self._id_mapping_lock = Lock()
        self._to_data_id = {}
        self._to_model_facing_id = {}
        self.model_facing_id_stream = IdStream()

        # Per-instance documents and seen ledger, guarded by the per-instance _documents_lock.
        self._documents_lock = Lock()
        self.documents = {}
        self.seen_ledger = {} # Lookup data_id -> list[Interval]

    def _add_mapping(self, data_id: str, model_facing_id: str):
        """Record ``data_id <-> model_facing_id`` in both directions. The
        caller must hold ``_id_mapping_lock``."""
        self._to_data_id[model_facing_id] = data_id
        self._to_model_facing_id[data_id] = model_facing_id

    def _validate_mapping(self):
        """Check that the two id maps are exact inverses of each other,
        raising ``ValueError`` listing every mismatch. A debugging aid: the
        maps only diverge if code outside this class mutates them."""
        with self._id_mapping_lock:
            self._validate_mapping_locked()

    def _validate_mapping_locked(self):
        """``_validate_mapping`` for callers already holding ``_id_mapping_lock``."""
        mismatches = []
        for model_facing_id, data_id in self._to_data_id.items():
            reverse = self._to_model_facing_id.get(data_id)
            if reverse != model_facing_id:
                mismatches.append(
                    f"_to_data_id[{model_facing_id!r}] = {data_id!r}, "
                    f"but _to_model_facing_id[{data_id!r}] = {reverse!r}"
                )
        for data_id, model_facing_id in self._to_model_facing_id.items():
            reverse = self._to_data_id.get(model_facing_id)
            if reverse != data_id:
                mismatches.append(
                    f"_to_model_facing_id[{data_id!r}] = {model_facing_id!r}, "
                    f"but _to_data_id[{model_facing_id!r}] = {reverse!r}"
                )
        if mismatches:
            raise ValueError(
                "id mappings are inconsistent:\n" + "\n".join(mismatches)
            )

    def to_data_id(self, model_facing_id: str) -> str:
        """The data_id (your database id) behind ``model_facing_id``. Global
        to the fork family, like the mapping itself. Raises ``KeyError`` for
        an id the family never minted — test model-provided ids with
        ``contains_model_facing_id`` instead of catching this."""
        try:
            return self._to_data_id[model_facing_id]
        except KeyError:
            raise KeyError(
                (f"model_facing_id {model_facing_id} not found in cache. use "
                "DocumentCache._validate_mapping() to check for inconsistencies.")
            ) from None

    def to_model_facing_id(self, data_id: str) -> str:
        """The model facing id minted for ``data_id``. Global to the fork
        family. Raises ``KeyError`` if no cache in the family has added the
        document."""
        try:
            return self._to_model_facing_id[data_id]
        except KeyError:
            raise KeyError(
                (f"data_id {data_id} not found in cache. use "
                "DocumentCache._validate_mapping() to check for inconsistencies.")
            ) from None

    def _mint_model_facing_id(self) -> str:
        """A short id not yet used by this cache or any cache forked from it."""
        return next(self.model_facing_id_stream)

    def add_document(self, data_id: str, document: Dict[str, Any]) -> str:
        """Add a document to this cache and return its model facing id (minted
        on first use, stable across forks sharing this cache's id mapping).

        Documents are add-once per cache instance: the seen ledger's character
        offsets index the content that was added first, so adding the same
        data_id again raises instead of silently replacing the document and
        resetting what the model has already been shown. Guard with
        ``data_id in cache`` when the same document can come back from
        several searches."""
        with self._documents_lock:
            if data_id in self.documents:
                raise ValueError(
                    f"data_id {data_id!r} is already in this cache; documents are add-once "
                    f"per cache instance — the seen ledger's offsets index the content that "
                    f"was added first"
                )
            self.documents[data_id] = document
            self.seen_ledger[data_id] = []

        with self._id_mapping_lock:

            if data_id in self._to_model_facing_id:
                model_facing_id = self.to_model_facing_id(data_id)
            else:
                model_facing_id = self._mint_model_facing_id()
                self._add_mapping(data_id, model_facing_id)

        return model_facing_id

    def update_seen(self, document_view: DocumentViewWithSnippet):
        """Record the character ranges ``document_view`` displayed into this
        cache's seen ledger, so later views of the same document collapse
        them to ``[seen: ...]`` markers. Call after rendering a view; a
        snippet-less view has no display spans and records nothing."""
        with self._documents_lock:
            seen_spans = self.seen_ledger[document_view.data_id]

            for char_span in document_view.snippet_display_spans:
                seen_spans = insert_interval(seen_spans, char_span)

            self.seen_ledger[document_view.data_id] = seen_spans

    @staticmethod
    def snippet_fn(query, content, snippet_size, language="english"):
        """The snippet selector behind ``apply_snippet``: the ``(start, end)``
        Python-character span of the ``snippet_size`` source-token stretch of
        ``content`` most relevant to ``query``. Unigram and bigram BM25 scores
        are calculated by the native Alyze selector. Override in a subclass
        to swap in a different selection strategy."""
        stride = stride_size(snippet_size)
        span = bm25_snippet_with_stride(
            query,
            content,
            window_size=snippet_size,
            stride=stride,
            language=language,
        )
        return span

    def apply_snippet(
        self,
        data_id: str,
        snippet_field: str,
        query: str,
        snippet_size: int = SNIPPET_SIZE_DEFAULT,
        min_seen_overlap: int = MIN_SEEN_OVERLAP_DEFAULT,
        display_fields: List[str] = None,
        language: str | None = None,
    ) -> DocumentView:
        """
        Apply a snippet to a document.
        snippet_size is the number of Alyze source tokens to include in the snippet.
        language overrides this cache's default analyzer language for this call.
        display_fields defaults to every field of the document dict, so leave
        your database id (and anything else private) out of the dict, or pass
        display_fields to display only a subset.
        """
        with self._documents_lock:
            document = self.documents[data_id]
            seen = self.seen_ledger[data_id]

        model_facing_id = self.to_model_facing_id(data_id)

        if display_fields is None:
            display_fields = document.keys()

        content = document[snippet_field]

        if snippet_field not in display_fields:
            raise ValueError(
                f"snippet field {snippet_field!r} must be in display_fields for document {data_id!r}"
            )
        if not isinstance(content, str) or not content:
            raise TypeError(
                f"snippet field {snippet_field!r} of document {data_id!r} must be a "
                f"non-empty string, got {content!r}"
            )

        effective_language = self.language if language is None else _validate_language(language)
        span = self.snippet_fn(query, content, snippet_size, effective_language)

        if overlap_list(span, seen) == [span]:
            snippet_seen_spans = [span]
            snippet_display_spans = []
        else:
            snippet_display_spans, snippet_seen_spans = plan_segments(span, seen, min_seen_overlap=min_seen_overlap)

        return DocumentViewWithSnippet(data_id, model_facing_id, document, snippet_field, snippet_seen_spans, snippet_display_spans, display_fields)

    def get_single_span_document_view(
        self,
        data_id: str,
        snippet_field: str = None,
        snippet_display_span: Tuple[int, int] = None,
        display_fields: List[str] = None
    ) -> DocumentView:
        """
        Specify snippet_field to return a DocumentViewWithSnippet (Ignores already seen spans).
        Leave snippet_display_span as None to render the entire snippet.
        display_fields defaults to every field of the document dict, so leave
        your database id (and anything else private) out of the dict, or pass
        display_fields to display only a subset.

        A span reaching past the end of the document is clamped to the
        document's length (the rendered ranged id names the clamped span, so
        it stays a verbatim-slice contract). A span lying entirely outside the
        document raises ``ValueError``: like every exception this class
        raises, it means a bug in the integration — model-provided spans are
        the tool layer's job to bounds-check (with the trained message) before
        building a view.
        """

        document = self.get_document(data_id)
        model_facing_id = self.to_model_facing_id(data_id)

        if display_fields is None:
            display_fields = document.keys()

        if snippet_field is None:
            return DocumentView(data_id, model_facing_id, document, display_fields)

        content = document[snippet_field]
        if not isinstance(content, str) or not content:
            raise TypeError(
                f"snippet field {snippet_field!r} of document {data_id!r} must be a "
                f"non-empty string, got {content!r}"
            )

        if snippet_display_span is None:
            snippet_display_span = (0, len(content))
        else:
            start, end = snippet_display_span
            if start >= end:
                raise ValueError(
                    f"empty span {start}:{end} for document {data_id!r} (start must be < end)"
                )
            if start >= len(content):
                raise ValueError(
                    f"span {start}:{end} lies entirely outside document {data_id!r} "
                    f"({len(content)} chars); bounds-check model-provided spans in "
                    f"your tool layer before building a view"
                )
            snippet_display_span = (max(0, start), min(end, len(content)))

        return DocumentViewWithSnippet(
            data_id,
            model_facing_id,
            document,
            snippet_field,
            [],
            [snippet_display_span],
            display_fields
        )

    def snap_char_range(self, data_id: str, snippet_field: str, char_range: Tuple[int, int]) -> Tuple[int, int]:
        """A model-requested ``char_range``, made displayable: clamped to the
        document, then edges snapped outward to word boundaries, exactly as
        trained. A range lying entirely outside the document returns the full
        document instead — a deliberate training-inference divergence that
        keeps every model-provided range renderable.

        Raises ``KeyError`` for an unknown ``data_id`` or missing
        ``snippet_field``: like every exception this class raises, that means
        a bug in the integration, not a model mistake."""
        content = self.get_document(data_id)[snippet_field]
        start, end = char_range
        if start >= len(content) or end <= 0:
            return 0, len(content)
        start, end = max(0, start), min(end, len(content))
        while start > 0 and not content[start - 1].isspace():
            start -= 1
        while end < len(content) and not content[end].isspace():
            end += 1
        return start, end

    def __contains__(self, data_id: str) -> bool:
        """True if this cache instance holds a document for ``data_id``.

        Membership is fork-local: forks copy ``documents``, so a document a
        sibling fork added after the split is not a member here even though
        the shared id mapping (one per fork family) may already know its
        data_id. Takes data_ids, like ``get_document`` — not model facing ids."""
        with self._documents_lock:
            return data_id in self.documents

    def contains_global(self, data_id: str) -> bool:
        """True if any cache in this fork family has added a document for
        ``data_id`` — i.e. it has a model facing id in the shared id mapping.

        Global membership does not imply local membership: a data_id added by
        a sibling fork is global but not ``in`` this instance, and its
        document is not readable here. Takes data_ids, like ``__contains__``."""
        with self._id_mapping_lock:
            return data_id in self._to_model_facing_id

    def contains_model_facing_id(self, model_facing_id: str) -> bool:
        """True if ``model_facing_id`` was minted by any cache in this fork
        family — i.e. ``to_data_id`` would succeed. The membership test for
        model-provided ids, so the tool layer can reject hallucinated ids
        without exception-driven control flow.

        Like ``contains_global`` (and unlike ``__contains__``), this spans the
        whole fork family via the shared id mapping: True does not imply the
        document is readable on *this* instance — a sibling fork may have
        added it. Takes model facing ids, not data_ids."""
        with self._id_mapping_lock:
            return model_facing_id in self._to_data_id

    def get_document(self, data_id: str) -> Dict[str, Any]:
        """The document dict this cache instance holds for ``data_id``.

        Raises ``KeyError`` if this instance never added the document, with a
        message distinguishing a data_id only a sibling fork added from one
        the fork family has never seen."""
        with self._documents_lock:
            if data_id not in self.documents:
                if data_id in self._to_model_facing_id:
                    raise KeyError(f"data_id {data_id} not found in the document cache for this instance, but is in the global id mapping")
                else:
                    raise KeyError(f"data_id {data_id} not found in the document cache")

            return self.documents[data_id]

    def get_document_from_model_facing_id(self, model_facing_id: str) -> Dict[str, Any]:
        """``get_document``, looked up by model facing id instead of data_id;
        either lookup's ``KeyError`` propagates."""
        return self.get_document(self.to_data_id(model_facing_id))

    def fork(self, n_forks: int) -> List['DocumentCache']:
        """Split off n_forks caches that share this cache's id space.

        Parent and forks keep one model_facing_id <-> data_id mapping (shared
        by reference, including the lock and id stream, so ids never collide
        or diverge across forks). Documents and the seen ledger are copied:
        each fork starts from this cache's current state but evolves
        independently afterwards.
        """
        with self._documents_lock:
            documents_snapshot = dict(self.documents)
            seen_snapshot = {k: list(v) for k, v in self.seen_ledger.items()}

        forks = []
        for _ in range(n_forks):
            child = DocumentCache(language=self.language)
            child._to_data_id = self._to_data_id
            child._to_model_facing_id = self._to_model_facing_id
            child._id_mapping_lock = self._id_mapping_lock
            child.model_facing_id_stream = self.model_facing_id_stream
            child.documents = copy.deepcopy(documents_snapshot)
            child.seen_ledger = {k: list(v) for k, v in seen_snapshot.items()}
            forks.append(child)
        return forks
