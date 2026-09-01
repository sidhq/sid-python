import copy
from threading import Lock
from typing import List, Dict, Tuple, Any

from .document_rendering import DocumentView, DocumentViewWithSnippet
from .id_stream import IdStream
from ._snippet import bm25_snippet_with_stride
from .ranges import (
    InvalidCharacterRange,
    RangeMode,
    resolve_char_range as _resolve_char_range,
    validate_range_mode,
)

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
    def __init__(
        self,
        language: str = "english",
        range_mode: RangeMode = "lenient",
    ):
        """Create a cache whose snippets use ``language`` by default.

        Named languages lowercase tokens and apply Alyze's language-specific
        stopword list. ``generic`` lowercases without removing stopwords.
        Individual ``apply_snippet`` calls may override this setting.
        """
        self.language = _validate_language(language)
        self.range_mode = validate_range_mode(range_mode)
        # Shared by reference across forks, guarded by the shared _mapping_lock.
        self._id_mapping_lock = Lock()
        self._to_data_id = {}
        self._to_model_facing_id = {}
        self.model_facing_id_stream = IdStream()

        # The document store is shared across forks. Seen state belongs to one fork.
        self._documents_lock = Lock()
        self.documents = {}
        self._seen_lock = Lock()
        self.seen_ledger = {}  # Lookup data_id -> list[Interval]

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

    def _validate_store(self):
        """Check that the shared document store and id mapping agree exactly."""
        with self._documents_lock:
            with self._id_mapping_lock:
                self._validate_mapping_locked()
                document_ids = set(self.documents)
                mapped_ids = set(self._to_model_facing_id)
                if document_ids != mapped_ids:
                    missing_documents = sorted(mapped_ids - document_ids)
                    missing_mappings = sorted(document_ids - mapped_ids)
                    details = []
                    if missing_documents:
                        details.append(
                            f"mapped ids without documents: {missing_documents!r}"
                        )
                    if missing_mappings:
                        details.append(
                            f"documents without mapped ids: {missing_mappings!r}"
                        )
                    raise ValueError(
                        "document store and id mapping are inconsistent: "
                        + "; ".join(details)
                    )

    def to_data_id(self, model_facing_id: str) -> str:
        """The data_id (your database id) behind ``model_facing_id``. Global
        to the fork family, like the mapping itself. Raises ``KeyError`` for
        an id the family never minted — test model-provided ids with
        ``contains_model_facing_id`` instead of catching this."""
        with self._id_mapping_lock:
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
        with self._id_mapping_lock:
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
        """Add a document and return its stable, fork-family model-facing id.

        The first insertion stores one deep copy. Re-adding a bound ``data_id``
        is an idempotent lookup and does not inspect or replace the stored copy.
        """
        with self._documents_lock:
            stored_document = None
            if data_id not in self.documents:
                stored_document = copy.deepcopy(document)
            with self._id_mapping_lock:
                model_facing_id = self._to_model_facing_id.get(data_id)
                if model_facing_id is None:
                    model_facing_id = self._mint_model_facing_id()
                if stored_document is not None:
                    self.documents[data_id] = stored_document
                if data_id not in self._to_model_facing_id:
                    self._add_mapping(data_id, model_facing_id)

        with self._seen_lock:
            self.seen_ledger.setdefault(data_id, [])

        return model_facing_id

    def update_seen(self, document_view: DocumentViewWithSnippet):
        """Record the character ranges ``document_view`` displayed into this
        cache's seen ledger, so later views of the same document collapse
        them to ``[seen: ...]`` markers. Call after rendering a view; a
        snippet-less view has no display spans and records nothing."""
        self.get_document(document_view.data_id)
        with self._seen_lock:
            seen_spans = self.seen_ledger.setdefault(document_view.data_id, [])

            for char_span in document_view.snippet_display_spans:
                seen_spans = insert_interval(seen_spans, char_span)

            self.seen_ledger[document_view.data_id] = seen_spans

    @staticmethod
    def snippet_fn(query, content, snippet_size, language="english"):
        """The snippet selector behind ``apply_snippet``: the ``(start, end)``
        Python-character span of the ``snippet_size`` source-token stretch of
        ``content`` most relevant to ``query``. Unigram and bigram BM25 scores
        are calculated by the native Alyze selector. Override in a subclass
        to swap in a different selection strategy. A query whose analyzed
        tokens are all stopwords gives every window a zero score and
        deliberately returns the beginning of the document."""
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
        document = self.get_document(data_id)
        with self._seen_lock:
            seen = list(self.seen_ledger.get(data_id, []))

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
        span = _resolve_char_range(
            span,
            len(content),
            mode="strict",
            document_id=data_id,
        )

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

        The configured range policy resolves a supplied span before rendering.
        The rendered ranged id always names the exact, verbatim Python slice.
        """

        document = self.get_document(data_id)
        model_facing_id = self.to_model_facing_id(data_id)

        if snippet_field is None:
            if display_fields is None:
                raise TypeError(
                    f"display_fields is required when snippet_field is None: the default "
                    f"(every field of document {data_id!r}) would render the full content "
                    f"inside an attribute of an empty <doc> tag. Pass the metadata fields "
                    f"to list, e.g. display_fields=['title', 'date']"
                )
            return DocumentView(data_id, model_facing_id, document, display_fields)

        if display_fields is None:
            display_fields = document.keys()

        content = document[snippet_field]
        if not isinstance(content, str) or not content:
            raise TypeError(
                f"snippet field {snippet_field!r} of document {data_id!r} must be a "
                f"non-empty string, got {content!r}"
            )

        if snippet_display_span is None:
            snippet_display_span = (0, len(content))
        else:
            snippet_display_span = self.resolve_char_range(
                data_id,
                snippet_field,
                snippet_display_span,
            )

        return DocumentViewWithSnippet(
            data_id,
            model_facing_id,
            document,
            snippet_field,
            [],
            [snippet_display_span],
            display_fields
        )

    def resolve_char_range(
        self,
        data_id: str,
        snippet_field: str,
        char_range: tuple[int, int],
    ) -> tuple[int, int]:
        """Resolve ``char_range`` to an exact half-open slice of a document."""
        content = self.get_document(data_id)[snippet_field]
        if not isinstance(content, str) or not content:
            raise TypeError(
                f"snippet field {snippet_field!r} of document {data_id!r} must be a "
                f"non-empty string, got {content!r}"
            )
        return _resolve_char_range(
            char_range,
            len(content),
            mode=self.range_mode,
            document_id=data_id,
        )

    def __contains__(self, data_id: str) -> bool:
        """True if this fork family holds a document for ``data_id``."""
        with self._documents_lock:
            return data_id in self.documents

    def contains_model_facing_id(self, model_facing_id: str) -> bool:
        """True if ``model_facing_id`` was minted by any cache in this fork
        family — i.e. ``to_data_id`` would succeed. The membership test for
        model-provided ids, so the tool layer can reject hallucinated ids
        without exception-driven control flow.

        A recognized id always names a document readable by every member of
        the fork family. Takes model-facing ids, not data ids."""
        with self._id_mapping_lock:
            return model_facing_id in self._to_data_id

    def get_document(self, data_id: str) -> Dict[str, Any]:
        """Return the fork-family document stored for ``data_id``."""
        with self._documents_lock:
            if data_id not in self.documents:
                raise KeyError(f"data_id {data_id} not found in the document cache")

            return self.documents[data_id]

    def get_document_from_model_facing_id(self, model_facing_id: str) -> Dict[str, Any]:
        """``get_document``, looked up by model facing id instead of data_id;
        either lookup's ``KeyError`` propagates."""
        return self.get_document(self.to_data_id(model_facing_id))

    @classmethod
    def _sibling(
        cls,
        source: 'DocumentCache',
        seen_ledger: Dict[str, List[Interval]],
    ) -> 'DocumentCache':
        """Build a fork-family sibling without allocating throwaway state."""
        child = cls.__new__(cls)
        child.language = source.language
        child.range_mode = source.range_mode
        child._to_data_id = source._to_data_id
        child._to_model_facing_id = source._to_model_facing_id
        child._id_mapping_lock = source._id_mapping_lock
        child.model_facing_id_stream = source.model_facing_id_stream
        child._documents_lock = source._documents_lock
        child.documents = source.documents
        child._seen_lock = Lock()
        child.seen_ledger = {key: list(spans) for key, spans in seen_ledger.items()}
        return child

    def fork(self, n_forks: int) -> List['DocumentCache']:
        """Create caches sharing documents and ids, with copied seen state."""
        with self._seen_lock:
            seen_snapshot = {k: list(v) for k, v in self.seen_ledger.items()}

        return [DocumentCache._sibling(self, seen_snapshot) for _ in range(n_forks)]
