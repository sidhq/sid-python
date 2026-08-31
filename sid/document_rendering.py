import html
from typing import List, Dict, Tuple, Any


# We model facing ids render as "model_facing_id#start:end"
ID_SEPARATOR = "#"
SPAN_SEPARATOR = ":"

def parse_rendered_model_facing_id(model_id_with_char_span):
    """Split a model-provided reference into ``(model_facing_id, char_span)``;
    ``char_span`` is ``None`` for a bare id. Every malformed shape raises
    ``ValueError`` carrying the trained message: the reference is model input,
    so your tool layer should catch ``ValueError`` around this call and
    re-raise it as its tool-error type, feeding the message back to the model
    as a failed tool call. The offsets are parsed with ``int()``, which is
    deliberately lenient about formatting; bounds against the document are the
    view's job, not the parser's."""
    if not isinstance(model_id_with_char_span, str) or not model_id_with_char_span:
        raise ValueError(
            f"invalid document reference {model_id_with_char_span!r}: expected a non-empty string"
        )
    if ID_SEPARATOR not in model_id_with_char_span:
        return model_id_with_char_span, None

    model_facing_id, _, char_span_str = model_id_with_char_span.partition(ID_SEPARATOR)
    if not model_facing_id:
        raise ValueError(
            f"invalid document reference {model_id_with_char_span!r}: missing document id before "
            f"'{ID_SEPARATOR}'"
        )
    try:
        char_span = tuple(int(x) for x in char_span_str.split(SPAN_SEPARATOR))
    except ValueError:
        char_span = ()
    if len(char_span) != 2:
        raise ValueError(
            f"invalid document reference {model_id_with_char_span!r}: expected "
            f"'<doc_id>{ID_SEPARATOR}<start>:<end>' with integer character "
            f"offsets, e.g. '{model_facing_id}{ID_SEPARATOR}10:70'"
        )
    start, end = char_span
    if start >= end:
        raise ValueError(
            f"invalid document reference {model_id_with_char_span!r}: range {start}:{end} is empty "
            f"(start must be < end)"
        )
    return model_facing_id, char_span


def _stringify(value) -> str:
    """A field value as display text; lists join with commas."""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _esc(value) -> str:
    """A field value made safe inside a ``<doc>`` tag (``&``, ``<``, ``>``
    escaped; quotes left alone)."""
    return html.escape(_stringify(value), quote=False)


def _md_cell(value) -> str:
    """A field value made safe inside a markdown table cell (pipes escaped,
    newlines flattened to spaces)."""
    return _stringify(value).replace("|", "\\|").replace("\n", " ")


def _seen_marker(char_span) -> str:
    """The ``[seen: "#start:end"]`` marker that stands in for an
    already-shown stretch of content."""
    start, end = char_span
    return f'[seen: "{ID_SEPARATOR}{start}:{end}"]'


class DocumentView:
    def __init__(
        self, data_id: str,
        model_facing_id: str,
        document: Dict[str, Any],
        display_fields: List[str]
    ):
        self.data_id = data_id
        self.model_facing_id = model_facing_id
        self.document = document
        self.display_fields = display_fields
        # A snippet-less view displays attribute fields only, never a span of
        # content — so update_seen has nothing to record for it.
        self.snippet_display_spans: List[Tuple[int, int]] = []

    def _render_parts(self, display_fields: List[str] = None) -> Tuple[str, Dict[str, Any], str]:
        """The pieces every render format shares: ``(doc_id, attrs, body)`` —
        the model-facing id (with a ``#start:end`` range where one applies),
        the attribute fields to show, and the snippet body (``None`` for a
        snippet-less view)."""
        fields = self.display_fields if display_fields is None else display_fields
        attrs = {field: self.document[field] for field in fields}
        return self.model_facing_id, attrs, None

    def render_xml(self) -> str:
        """Render this view as the model-facing string: a ``<doc>`` tag whose
        id names the exact character span shown, the display fields as
        attributes, and the snippet text (with any ``[seen: ...]`` markers)
        as the body. A snippet-less view renders as an empty tag."""
        doc_id, attrs, body = self._render_parts()
        parts = [f'id="{_esc(doc_id)}"']

        for key, value in attrs.items():
            if value:
                parts.append(f"{key}={value}" if isinstance(value, int) else f'{key}="{_esc(value)}"')

        if body is None:
            return f'<doc {" ".join(parts)}></doc>'
        else:
            return f'<doc {" ".join(parts)}>\n{body}\n</doc>'


class DocumentViewWithSnippet(DocumentView):
    def __init__(
        self, data_id: str,
        model_facing_id: str,
        document: Dict[str, Any],
        snippet_field: str,
        snippet_seen_spans: List[Tuple[int, int]],
        snippet_display_spans: List[Tuple[int, int]],
        display_fields: List[str]
    ):
        super().__init__(data_id, model_facing_id, document, display_fields)
        self.snippet_field = snippet_field
        self.snippet_seen_spans = snippet_seen_spans
        self.snippet_display_spans = snippet_display_spans

    def _render_parts(self, display_fields: List[str] = None) -> Tuple[str, Dict[str, Any], str]:
        fields = self.display_fields if display_fields is None else display_fields
        content = self.document[self.snippet_field]
        segments = sorted(
            [("show", x, y) for x, y in self.snippet_display_spans]
            + [("seen", x, y) for x, y in self.snippet_seen_spans],
            key=lambda segment: segment[1],
        )
        a, b = segments[0][1], segments[-1][2]
        fully_seen = not self.snippet_display_spans

        parts = []

        if a > 0 and not fully_seen:
            parts.append("... ")

        for kind, x, y in segments:
            parts.append(content[x:y] if kind == "show" else _seen_marker((x, y)))

        if b < len(content) and not fully_seen:
            parts.append(" ...")

        body = "".join(parts)

        doc_id = self.model_facing_id
        if not fully_seen and (a, b) != (0, len(content)):
            doc_id += f"{ID_SEPARATOR}{a}{SPAN_SEPARATOR}{b}"

        attrs = {"doc_length": len(content)}
        for field in fields:
            if field != self.snippet_field:
                attrs[field] = self.document[field]
        return doc_id, attrs, body


def render_markdown_table(document_views: List[DocumentView], display_fields: List[str] = None) -> str:
    """Render ``document_views`` as one markdown table, one row per view,
    with the same ids and field content as ``render_xml`` — more
    token-efficient than a run of ``<doc>`` tags for tabular-like results.

    Columns are ``id``, any per-view extras (such as ``doc_length``), then
    ``display_fields``, which defaults to the first view's. Views missing a
    column render an empty cell."""
    if not document_views:
        return ""
    if display_fields is None:
        display_fields = document_views[0].display_fields
    display_fields = list(display_fields)

    rows = []
    for view in document_views:
        doc_id, attrs, body = view._render_parts(display_fields)
        cells = {"id": doc_id, **attrs}
        if body is not None:
            cells[view.snippet_field] = body
        rows.append(cells)

    columns = ["id"] + [key for key in rows[0] if key != "id" and key not in display_fields] + display_fields
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("-" * (len(column) + 2) for column in columns) + "|",
    ]
    for cells in rows:
        lines.append("| " + " | ".join(_md_cell(cells.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)
