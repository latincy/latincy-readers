# CombinedReader Design

**Date:** 2026-02-11
**Status:** Approved
**Target:** latincy-readers v2

## Summary

A `CombinedReader` class that merges outputs from multiple corpus readers through a unified interface. Uses `itertools.chain` to lazily combine iterator-based methods and dict merging for aggregate methods.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| API scope | Full common interface (core + search/analysis) | Feature-rich v2 goal |
| Fileid namespacing | Prefix with reader class name | Readable, self-documenting |
| Protocol conformance | Does NOT implement `CorpusReaderProtocol` | Compositor, not a reader; `root` property doesn't apply |
| Citation-specific methods | Not included | Format-dependent; users access via individual readers |
| `concordance()` merging | Merge dicts, concatenate citation lists | Matches "unified view" philosophy |
| `FileSelector` support | Not included | Too coupled to single-reader metadata stores |

## Class Structure

### Constructor

```python
class CombinedReader:
    def __init__(self, *readers, prefixes: dict | None = None):
        ...
```

Accepts three styles:

```python
# Auto-prefix from class name (TesseraeReader -> "tesserae")
CombinedReader(tess_reader, ll_reader)

# Explicit prefix via tuple
CombinedReader(("tess", tess_reader), ("ll", ll_reader))

# Mixed
CombinedReader(tess_reader, ("ll", ll_reader))
```

Auto-prefix derivation: lowercase class name, strip trailing "reader".

### File ID Namespacing

`fileids()` returns namespaced IDs: `"tesserae/vergil.aeneid.txt"`.

All methods accepting `fileids` parse the prefix to route to the correct reader via `_resolve_fileids()`.

### Core Iteration Methods

All chain across readers via `itertools.chain.from_iterable`:

- `docs(fileids=None) -> Iterator[Doc]`
- `sents(fileids=None, as_text=False) -> Iterator[Span | str]`
- `tokens(fileids=None, as_text=False) -> Iterator[Token | str]`
- `texts(fileids=None) -> Iterator[str]`
- `metadata(fileids=None) -> Iterator[tuple[str, dict]]` (namespaces fileids in output)

### Search & Analysis Methods

Iterator-based methods chain; `concordance()` merges dicts:

- `find_sents(fileids=None, **kwargs) -> Iterator[dict]`
- `kwic(keyword, fileids=None, **kwargs) -> Iterator[dict[str, str]]`
- `ngrams(n=2, fileids=None, **kwargs) -> Iterator[str | tuple]`
- `skipgrams(n=2, k=1, fileids=None, **kwargs) -> Iterator[str | tuple]`
- `concordance(fileids=None, **kwargs) -> dict[str, list[str]]` (merged across readers)

### Properties

- `readers -> dict[str, CorpusReaderProtocol]` — access individual readers by prefix

### Convenience Function

```python
def combine(*readers, **kwargs) -> CombinedReader:
    """Shorthand for CombinedReader(*readers)."""
    return CombinedReader(*readers, **kwargs)
```

## File Location

- Implementation: `src/latincyreaders/core/combined.py`
- Tests: `tests/test_combined_reader.py`
- Exports: added to `src/latincyreaders/__init__.py`

## Future Roadmap (v2 subrelease)

### Citation-Aware Methods
- `combined.citation_readers` property filtering to `CitationReaderProtocol` conformers
- `combined.lines()` / `combined.doc_rows()` chaining only across citation-aware readers

### Cross-Corpus FileSelector
- `combined.select()` querying metadata across all readers
- Namespaced fileid support in FileSelector

### Deduplication
- `dedupe=True` flag using metadata matching (author + work + section)
- Reader list order = priority (first reader wins ties)
- Likely driven by FileSelector-style API: `combined.select().dedupe(on=["author", "work"])`
