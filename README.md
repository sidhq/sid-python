# SID SDK

`sid-sdk` turns search results into compact, model-facing document views. It
assigns stable short IDs, selects relevant snippets, tracks the exact character
ranges already shown, masks repeated text, and renders SID's `<doc>` format.

Snippet selection is implemented in Rust with
[Alyze](https://github.com/turbopuffer/alyze) Unicode UAX #29 tokenization and
separate unigram and bigram BM25 scores. Returned ranges are Python character
offsets into the original text.

## Installation

```console
pip install sid-sdk
```

Prebuilt ABI3 wheels require no Rust compiler on supported macOS, Linux, and
Windows platforms. Building from the source distribution requires Python 3.12+
and Rust 1.88 or newer.

## Basic use

```python
from sid import DocumentCache

cache = DocumentCache(language="english")
cache.add_document(
    "database-id",
    {"title": "Example", "content": "The complete document text ..."},
)
view = cache.apply_snippet(
    "database-id",
    snippet_field="content",
    query="complete document",
)
print(view.render_xml())
cache.update_seen(view)
```

Named analyzers are available for Danish, Dutch, English, Finnish, French,
German, Hungarian, Italian, Norwegian, Portuguese, Russian, Spanish, and
Swedish. They lowercase tokens and remove Alyze's language-specific stopwords.
Use `language="generic"` for lowercase Unicode tokenization without stopword
removal. Chinese, Japanese, and Korean text is handled through `generic` UAX
#29 segmentation; there are no dedicated language entries. `apply_snippet`
accepts a one-call `language=` override, and forks inherit the cache default.

If every query token is a stopword, all windows have the same score and the
selector deliberately returns the beginning of the document.

## Model-facing IDs

New caches assign six ASCII digit strings, from `"000000"` through `"999999"`,
instead of the previous five lowercase letters. IDs remain strings: preserve
leading zeros in storage, JSON tool schemas, and model responses. For example,
`"004271"` and the ranged reference `"004271#10:70"` identify the same document.
Database IDs are unchanged. Parsers still accept older letter-based references;
resolving any reference requires its original cache mapping.

Each cache family can mint 1,000,000 distinct IDs over its lifetime, down from
11,881,376. Exhaustion raises `IdSpaceExhausted` without recycling IDs. Forks
share the stream and mappings; independent caches can assign the same ID to
different documents, so keep references scoped to their cache family. Check
model-provided IDs with `cache.contains_model_facing_id` before using them.

This default prioritizes reference accuracy based on observed model behavior.
Token cost depends on the tokenizer: the repository's historical measurements
were about six tokens per numeric ID on Qwen and two on o200k/cl100k, versus
about three for five-letter IDs. Downstream letter-only validators, prompt
examples, and seeded snapshots need updating; the same seed now produces a
different default sequence. Explicit custom `IdStream(alphabet=..., length=...)`
settings continue to work, including the old lowercase alphabet with length 5.

## Character ranges

`DocumentCache(range_mode="lenient")` is the default. A partially overlapping
range is reduced to its exact intersection with the document, while an empty,
inverted, malformed, or wholly disjoint range raises `InvalidCharacterRange`.
Use `range_mode="strict"` to require `0 <= start < end <= len(content)` exactly.
Ranges are half-open Python character offsets and are never expanded to word
boundaries, reversed, or replaced with the complete document.

Call `cache.resolve_char_range(data_id, snippet_field, (start, end))` before
passing a model-provided range onward. `get_single_span_document_view` applies
the same configured policy, and rendered IDs always describe the exact slice
displayed.

## Rendering and forks

XML document bodies and attribute values escape `&`, `<`, and `>`; quotes are
left unchanged. Falsy attributes are omitted from XML. Markdown tables stringify
their values normally, so empty strings make empty cells while `0` and `False`
remain visible. Snippet-less views require explicit `display_fields`.

Forks share one document store, ID mapping, and ID stream. A document added by
any family member is immediately readable by the others, and adding the same
`data_id` again is an idempotent lookup. Each fork keeps its own seen ledger,
initialized from the parent's current ledger, so agents can track what they
have shown independently.

Concurrent document additions are safe across a fork family, and separate
forks may run on separate threads. A single fork is intended for one agent;
`apply_snippet` and `update_seen` are separate ledger operations and are not an
atomic read-modify-write pair when called concurrently on the same fork.

## Development

```console
uv sync --group dev
uv run maturin develop --locked
uv run pytest -rxX --strict-markers
cargo test --locked
```

## License

MIT. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
