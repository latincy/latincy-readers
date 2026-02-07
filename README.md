<p align="center">
  <img src="assets/latincy-readers-logo.jpg" alt="LatinCy Readers" width="400">
</p>

# LatinCy Readers

Corpus readers for Latin and Ancient Greek texts with [LatinCy](https://github.com/diyclassics/latincy) and [OdyCy](https://centre-for-humanities-computing.github.io/odyCy/) integration.

Version 1.1.0; Python 3.10+; LatinCy 3.8.0+

## Installation

```bash
# Install from PyPI
pip install latincy-readers

# Install the LatinCy model (for Latin texts)
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.8.0-py3-none-any.whl

# Install the OdyCy model (for Ancient Greek texts)
pip install https://huggingface.co/chcaa/grc_odycy_joint_lg/resolve/main/grc_odycy_joint_lg-any-py3-none-any.whl

# For development (editable install)
git clone https://github.com/diyclassics/latincy-readers.git
cd latincy-readers
pip install -e ".[dev]"
```

## Quick Start

```python
from latincyreaders import TesseraeReader, AnnotationLevel

# Auto-download corpus on first use
reader = TesseraeReader()

# Or specify a custom path
reader = TesseraeReader("/path/to/tesserae/corpus")

# Iterate over documents as spaCy Docs
for doc in reader.docs():
    print(f"{doc._.fileid}: {len(list(doc.sents))} sentences")

# Search for sentences containing specific forms
for result in reader.find_sents(forms=["Caesar", "Caesarem"]):
    print(f"{result['citation']}: {result['sentence']}")

# Get raw text (no NLP processing)
for text in reader.texts():
    print(text[:100])
```

## Readers

| Reader | Format | Auto-Download | Description |
|--------|--------|---------------|-------------|
| `TesseraeReader` | `.tess` | Yes | CLTK Latin Tesserae corpus |
| `GreekTesseraeReader` | `.tess` | Yes | CLTK Greek Tesserae corpus (OdyCy) |
| `PlaintextReader` | `.txt` | No | Plain text files |
| `LatinLibraryReader` | `.txt` | Yes | Latin Library corpus |
| `TEIReader` | `.xml` | No | TEI-XML documents |
| `PerseusReader` | `.xml` | No | Perseus Digital Library TEI |
| `CamenaReader` | `.xml` | Yes | CAMENA Neo-Latin corpus |
| `TxtdownReader` | `.txtd` | No | Txtdown format with citations |
| `UDReader` | `.conllu` | No | Universal Dependencies CoNLL-U |
| `LatinUDReader` | `.conllu` | Yes | All 6 Latin UD treebanks |

### Auto-Download

Readers with auto-download support will automatically fetch the corpus on first use:

```python
# Downloads to ~/latincy_data/lat_text_tesserae/texts if not found
reader = TesseraeReader()

# Disable auto-download
reader = TesseraeReader(auto_download=False)

# Use environment variable for custom location
# export TESSERAE_PATH=/custom/path
reader = TesseraeReader()

# Manual download to specific location
TesseraeReader.download("/path/to/destination")
```

### Ancient Greek (GreekTesseraeReader)

Read Ancient Greek texts from the CLTK Greek Tesserae corpus using OdyCy NLP models:

```python
from latincyreaders import GreekTesseraeReader, AnnotationLevel

# Auto-download Greek Tesserae corpus on first use
reader = GreekTesseraeReader()

# Use TOKENIZE level (no OdyCy model needed)
reader = GreekTesseraeReader(annotation_level=AnnotationLevel.TOKENIZE)

# Iterate over citation lines
for citation, text in reader.texts_by_line():
    print(f"{citation}: {text[:60]}...")

# Search for Greek words
for fid, cit, text, matches in reader.search(r"Ἀχιλ"):
    print(f"{cit}: found {matches}")

# Environment variable for custom location
# export GRC_TESSERAE_PATH=/custom/path
reader = GreekTesseraeReader()
```

### Universal Dependencies Treebanks

Access gold-standard linguistic annotations from Latin UD treebanks:

```python
from latincyreaders import LatinUDReader, PROIELReader

# See available treebanks
LatinUDReader.available_treebanks()
# {'proiel': 'Vulgate, Caesar, Cicero, Palladius',
#  'perseus': 'Classical texts from Perseus Digital Library',
#  'ittb': 'Index Thomisticus (Thomas Aquinas)',
#  'llct': 'Late Latin Charter Treebank',
#  'udante': "Dante's Latin works",
#  'circse': 'CIRCSE Latin treebank'}

# Use a specific treebank
reader = PROIELReader()

# Iterate sentences with UD annotations
for sent in reader.ud_sents():
    print(f"{sent._.citation}: {sent.text}")

# Access full UD token data
for token in doc:
    ud = token._.ud  # dict with all 10 CoNLL-U columns
    print(f"{token.text}: {ud['upos']} {ud['feats']}")

# Read from all treebanks at once
reader = LatinUDReader()
LatinUDReader.download_all()  # Download all 6 treebanks
```

**Note:** Unlike other readers, `UDReader` constructs spaCy Docs directly from gold UD annotations rather than running the spaCy NLP pipeline.

## Core API

All readers provide a consistent interface:

```python
reader.fileids()              # List available files
reader.texts(fileids=...)     # Raw text strings (generator)
reader.docs(fileids=...)      # spaCy Doc objects (generator)
reader.sents(fileids=...)     # Sentence spans (generator)
reader.tokens(fileids=...)    # Token objects (generator)
reader.metadata(fileids=...)  # File metadata (generator)
```

### FileSelector: Fluent File Filtering

Use the `select()` method for complex file queries combining filename patterns and metadata:

```python
# Filter by filename pattern (regex)
vergil_docs = reader.select().match(r"vergil\..*")

# Filter by metadata
epics = reader.select().where(genre="epic")

# Multiple conditions (AND)
vergil_epics = reader.select().where(author="Vergil", genre="epic")

# Match any of multiple values
major_authors = reader.select().where(author__in=["Vergil", "Ovid", "Horace"])

# Date ranges
augustan = reader.select().date_range(-50, 50)

# Chain multiple filters
selection = (reader.select()
    .match(r".*aen.*")
    .where(genre="epic")
    .date_range(-50, 50))

# Use with docs(), sents(), etc.
for doc in reader.docs(selection):
    print(doc._.fileid)

# Preview results
print(selection.preview(5))
print(f"Found {len(selection)} files")
```

### Search API

```python
# Fast regex search (no NLP)
reader.search(pattern=r"\bbell\w+")

# Form-based sentence search
reader.find_sents(forms=["amor", "amoris"])

# Lemma-based search (requires NLP)
reader.find_sents(lemma="amo")

# spaCy Matcher patterns
reader.find_sents(matcher_pattern=[{"POS": "ADJ"}, {"POS": "NOUN"}])
```

### Text Analysis

```python
# Build a concordance (word -> citations mapping)
conc = reader.concordance(basis="lemma")
print(conc["amor"])  # ['<catull. 1.1>', '<verg. aen. 4.1>', ...]

# Keyword in Context
for hit in reader.kwic("amor", window=5, by_lemma=True):
    print(f"{hit['left']} [{hit['match']}] {hit['right']}")
    print(f"  -- {hit['citation']}")

# N-grams
for ngram in reader.ngrams(n=2, basis="lemma"):
    print(ngram)  # "qui do", "do lepidus", ...

# Skip-grams (n-grams with gaps)
for sg in reader.skipgrams(n=2, k=1):
    print(sg)
```

### Document Caching

Documents are cached by default for better performance when accessing the same file multiple times:

```python
# Caching enabled by default
reader = TesseraeReader()

# Disable caching
reader = TesseraeReader(cache=False)

# Configure cache size
reader = TesseraeReader(cache_maxsize=256)

# Check cache statistics
print(reader.cache_stats())  # {'hits': 5, 'misses': 3, 'size': 3, 'maxsize': 128}

# Clear the cache
reader.clear_cache()
```

### Annotation Levels

All linguistic annotations are provided by [LatinCy](https://github.com/diyclassics/latincy) spaCy-based pipelines. The full pipeline provides POS tagging, lemmatization, morphological analysis, and named entity recognition—but this can be slow for large corpora. If you don't need all annotations, you can get significant performance gains by selecting a lighter annotation level:

```python
from latincyreaders import AnnotationLevel

# Full pipeline: POS, lemma, morphology, NER (default)
reader = TesseraeReader(annotation_level=AnnotationLevel.FULL)

# Basic: tokenization + sentence boundaries only
reader = TesseraeReader(annotation_level=AnnotationLevel.BASIC)

# Tokenization only (no sentence boundaries)
reader = TesseraeReader(annotation_level=AnnotationLevel.TOKENIZE)

# No NLP at all - use texts() for raw strings
for text in reader.texts():
    print(text)
```

### Metadata Management

```python
from latincyreaders import MetadataManager, MetadataSchema

# Load and merge metadata from JSON files
manager = MetadataManager("/path/to/corpus")

# Access metadata
meta = manager.get("vergil.aen.tess")
print(meta["author"], meta["date"])

# Filter files by metadata
for fileid in manager.filter_by(author="Vergil", genre="epic"):
    print(fileid)

# Date range filtering
for fileid in manager.filter_by_range("date", -50, 50):
    print(fileid)

# Validate metadata against a schema
schema = MetadataSchema(
    required={"author": str, "title": str},
    optional={"date": int, "genre": str}
)
manager = MetadataManager("/path/to/corpus", schema=schema)
result = manager.validate()
if not result.is_valid:
    print(result.errors)
```

## Corpora Supported

- [Tesserae Latin Corpus](https://github.com/cltk/lat_text_tesserae)
- [Tesserae Greek Corpus](https://github.com/cltk/grc_text_tesserae)
- [Perseus Digital Library TEI](https://www.perseus.tufts.edu/)
- [Latin Library](https://github.com/cltk/lat_text_latin_library)
- [CAMENA Neo-Latin](https://github.com/nevenjovanovic/camena-neolatinlit)
- [Universal Dependencies Latin Treebanks](https://universaldependencies.org/) (PROIEL, Perseus, ITTB, LLCT, UDante, CIRCSE)
- Any plaintext, TEI-XML, or CoNLL-U collection

## CLI Tools

Search tool in `cli/`:

```bash
# Lemma search (slower, finds all inflected forms)
python cli/reader_search.py --lemmas Caesar --limit 100
python cli/reader_search.py --lemmas bellum pax --fileids "cicero.*"

# Form search (fast, exact match)
python cli/reader_search.py --forms Caesar Caesarem --limit 100

# Pattern search (fast, regex)
python cli/reader_search.py --pattern "\\bTheb\\w+" --output thebes.tsv
```

---

## Bibliography

- Bird, S., E. Loper, and E. Klein. 2009. *Natural Language Processing with Python*. O'Reilly: Sebastopol, CA.
- Bengfort, Benjamin, Rebecca Bilbro, and Tony Ojeda. 2018. *Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning*. O'Reilly: Sebastopol, CA.

---

*Developed by [Patrick J. Burns](http://github.com/diyclassics) with Claude Opus 4.5. in January 2026.*  
