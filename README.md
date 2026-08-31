# SID SDK

`sid-sdk` provides the context-management layer used to turn search results
into compact, model-facing document views. It assigns stable short document
IDs, selects relevant snippets, tracks the exact character ranges already
shown, masks repeated text, and renders the trained `<doc>` format.

Snippet selection is implemented in Rust using
[Alyze](https://github.com/turbopuffer/alyze) Unicode UAX #29 tokenization.
It ranks fixed source-token windows with separate unigram and bigram BM25
streams, so exact phrases receive evidence without preventing single-term
matches. Returned ranges are Python character offsets into the original text.

## Installation

```console
pip install sid-sdk
```

Prebuilt ABI3 wheels require no Rust compiler on supported macOS, Linux, and
Windows platforms. Linux wheels use portable manylinux tags rather than being
tied to one distribution, so the same wheel works across current Debian,
Ubuntu, and other compatible glibc-based distributions. Building the source
distribution requires Python 3.12+ and Rust 1.88 or newer.

Published wheels cover macOS (Apple Silicon and Intel), Windows x86-64,
manylinux2014 (x86-64 and ARM64), and musllinux 1.2 (x86-64 and ARM64). CI also
installs the manylinux wheel in supported Debian and Ubuntu releases; separate
per-distribution wheels would be byte-for-byte redundant with Python's Linux
wheel compatibility model.

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

`DocumentCache` accepts `danish`, `dutch`, `english`, `finnish`, `french`,
`german`, `hungarian`, `italian`, `norwegian`, `portuguese`, `russian`,
`spanish`, and `swedish`. These analyzers lowercase and remove the matching
Alyze stopwords. Use `generic` for lowercase Unicode tokenization without
stopword removal. Pass `language=` to `apply_snippet` to override the cache
default for one call; forks inherit the default.

## Development

```console
uv sync --group dev
uv run maturin develop --locked
uv run pytest -rxX --strict-markers
cargo test --locked
```

The two strict xfails in the current suite document pre-existing rendering
format differences. They are intentionally not snippet-selector tests.

## License

MIT. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
