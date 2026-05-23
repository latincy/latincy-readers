<p align="center">
  <img src="assets/latincy-readers-logo.jpg" alt="LatinCy Readers" width="400">
</p>

# LatinCy Readers

Corpus readers for Latin and Ancient Greek texts with [LatinCy](https://github.com/diyclassics/latincy) NLP integration.

Version 1.5.0; Python 3.10+; LatinCy 3.9.0+

## Installation

```bash
# Install the package
pip install latincy-readers

# With sentence vector search support
pip install latincy-readers[vectors]

# For development (editable install)
git clone https://github.com/latincy/latincy-readers.git
cd latincy-readers
pip install -e ".[dev]"
```

### Models

LatinCy NLP models are hosted on Hugging Face and installed separately (mirroring spaCy's pattern for language models). Install whichever you need:

```bash
# Latin model (la_core_web_lg)
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.9.0-py3-none-any.whl

# Ancient Greek model (grc_dep_web_lg)
pip install https://huggingface.co/latincy/grc_dep_web_lg/resolve/main/grc_dep_web_lg-3.8.1-py3-none-any.whl
```

You can skip model installation if you only need raw text iteration or `AnnotationLevel.TOKENIZE`.

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
| `GreekTesseraeReader` | `.tess` | Yes | CLTK Greek Tesserae corpus |
| `PlaintextReader` | `.txt` | No | Plain text files |
| `LatinLibraryReader` | `.txt` | Yes | Latin Library corpus |
| `TEIReader` | `.xml` | No | TEI-XML documents |
| `PerseusReader` | `.xml` | No | Perseus Digital Library TEI |
| `CamenaReader` | `.xml` | Yes | CAMENA Neo-Latin corpus |
| `DigilibLTReader` | `.xml` | No | digilibLT Late-Antique Latin TEI corpus |
| `PTAReader` | `.xml` | Yes | Patristic Text Archive (Greek & Latin) |
| `CSELReader` | `.xml` | No | Corpus Scriptorum Ecclesiasticorum Latinorum |
| `FormulaeReader` | `.xml` | No | Formulae-Litterae-Chartae medieval charters |
| `EpistolaeReader` | `.html.md` | No | Epistolae medieval women's Latin letters |
| `EDHReader` | `.xml` | Yes (prompt) | Epigraphic Database Heidelberg inscriptions |
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

Read Ancient Greek texts from the CLTK Greek Tesserae corpus using LatinCy Greek NLP models:

```python
from latincyreaders import GreekTesseraeReader, AnnotationLevel

# Auto-download Greek Tesserae corpus on first use
reader = GreekTesseraeReader()

# Use TOKENIZE level (no Greek model needed)
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

### digilibLT (DigilibLTReader)

Read TEI-XML files from [digilibLT](http://digiliblt.uniupo.it) (Digital Library of Late-Antique Latin Texts). Handles the structural variation found across the collection — flat `<p>` paragraphs, `<div type="cap">` chapters, nested `<div type="lib">` → `<div type="cap">` book/chapter hierarchies, single `<div type="section">` with `<head>` elements, and verse `<lg>/<l>` line groups — and exposes chapter-level structure as named spans:

```python
from latincyreaders import DigilibLTReader

reader = DigilibLTReader("/path/to/digilibt/xml")

# Rich metadata from teiHeader: DLT ID, author (via persName), source, creation date
for meta in reader.headers():
    print(meta["dlt_id"], meta.get("author"), meta.get("title"))

# Chapter-aware iteration — Spans with citations
for ch in reader.chapters():
    print(f"{ch._.citation}: {ch.text[:60]}...")

# Or as (citation, text) tuples without NLP overhead
for citation, text in reader.chapters(as_text=True):
    print(f"{citation}: {text[:60]}...")

# Chapter spans are also attached to each Doc
for doc in reader.docs():
    for ch in doc.spans.get("chapters", []):
        print(ch._.citation)
```

**Text-critical symbols.** With `use_symbols=True` (default), the reader strips editorial marks before NLP processing — `<supplied>` → `supplied`, `[secluded]` removed, `{corrected}` → `corrected`, `†crux†` → `crux`, `***` lacuna markers removed, and `M(arcus)` → `Marcus` abbreviation expansion. Set `use_symbols=False` to preserve the marks verbatim.

**License:** digilibLT texts are released under CC BY-NC-SA.

### Patristic Text Archive (PTAReader)

Read TEI-XML files from the [Patristic Text Archive](https://pta.bbaw.de) (Berlin-Brandenburg Academy of Sciences), which provides open-access ancient Christian texts in Greek and Latin under CC-BY 4.0.

The corpus contains ~210 Greek texts (~2.3M tokens) and a growing Latin collection. Each file yields one Doc per `<div type="textpart">` section, with CTS URN, language, author, title, and per-section citation in metadata.

```python
from latincyreaders import PTAReader, AnnotationLevel

# Auto-downloads to ~/latincy_data/pta_data on first use
reader = PTAReader()

# Or point at a local checkout
reader = PTAReader("/path/to/pta_data/data")

# Filter by language
for doc in reader.docs(fileids="*lat*.xml"):
    meta = doc._.metadata
    print(f"{meta['urn']} §{meta['citation']}: {doc.text[:80]}")

# Use AnnotationLevel.TOKENIZE for fast iteration without full NLP
reader = PTAReader(
    "/path/to/pta_data/data",
    annotation_level=AnnotationLevel.TOKENIZE,
)
texts = list(reader.texts(fileids="*grc*.xml"))
```

**Metadata per Doc:**

| Key | Example | Description |
|-----|---------|-------------|
| `urn` | `urn:cts:pta:pta0001.pta014.pta-lat1` | Work-level CTS URN |
| `language` | `lat` or `grc` | Language code |
| `author` | `Severianus Gabalensis` | Author from teiHeader |
| `title` | `In illud: Pone manum tuam` | Work title |
| `div_type` | `section` | Textpart subtype |
| `div_n` | `1` | Section number |
| `citation` | `1` | Human-readable section reference |

**License:** PTA texts are CC-BY 4.0 (per-file, see `<availability>` in each header).

### CSEL (CSELReader)

Read TEI-XML files from the [Corpus Scriptorum Ecclesiasticorum Latinorum](https://github.com/OpenGreekAndLatin/csel-dev) digital edition published by the Open Greek and Latin Project. The corpus contains Latin patristic and ecclesiastical texts encoded with a two-level `book`/`section` hierarchy.

Each file yields one Doc with `doc.spans["chapters"]` containing Span objects for every `<div subtype="section">`. Citations follow the form `"book 1, section 3"`.

```python
from latincyreaders import CSELReader, AnnotationLevel

# Point at a local clone of csel-dev
reader = CSELReader("/path/to/csel-dev/data")

# Iterate chapters with citations
for doc in reader.docs():
    meta = doc._.metadata
    print(f"{meta['author']}: {meta['title']} ({meta['cts_urn']})")
    for ch in doc.spans["chapters"]:
        print(f"  {ch._.citation}: {ch.text[:80]}")

# Fast header scan (no NLP)
for h in reader.headers():
    print(h["author"], h["cts_urn"])

# Raw (citation, text) pairs with zero NLP overhead
for citation, text in reader.chapters(as_text=True):
    print(citation, text[:60])
```

**Metadata per Doc:**

| Key | Example | Description |
|-----|---------|-------------|
| `author` | `Augustine` | Author from titleStmt |
| `title` | `Confessiones` | Latin title (`xml:lang="lat"`) |
| `cts_urn` | `urn:cts:latinLit:stoa0040.stoa001.opp-lat1` | CTS URN |
| `filename` | `stoa0040.stoa001.opp-lat1.xml` | Source filename |

**Chapter spans** (`doc.spans["chapters"]`): each Span carries `span._.citation` in the form `"book N, section N"`.

**License:** CC-BY-SA 4.0.

### Formulae-Litterae-Chartae (FormulaeReader)

Read TEI-XML files from the [Formulae-Litterae-Chartae](https://github.com/Formulae-Litterae-Chartae/formulae-open) project (University of Hamburg). The corpus covers early medieval Latin charters and formularies (500–1000 CE), with text stored as `<w>` (word) elements.

File pattern `**/*.lat*.xml` naturally excludes `__capitains__.xml` catalog files. French regest (`<front>`) is excluded; only the Latin `<div type="edition">` is extracted.

```python
from latincyreaders import FormulaeReader

reader = FormulaeReader("/path/to/formulae-open")

# Iterate over charters
for text in reader.texts():
    print(text[:100])

# Metadata per charter
for h in reader.headers():
    print(h["collection"], h["cts_urn"], h["date"])
```

**Metadata per Doc:**

| Key | Example | Description |
|-----|---------|-------------|
| `cts_urn` | `urn:cts:formulae:redon.courson0001.lat001` | CTS URN |
| `collection` | `redon` | Collection prefix from URN |
| `title` | `Cartulaire de Redon` | Document title |
| `date` | `832` | Date from `<dateline>` (leading zeros stripped) |
| `filename` | `redon.courson0001.lat001.xml` | Source filename |

**License:** CC-BY 4.0.

### Epistolae (EpistolaeReader)

Read Hugo Markdown (`.html.md`) files from the [Epistolae](https://github.com/ccnmtl/epistolae-hugo) project (Columbia University / University of Siena). The corpus contains ~1,100 medieval Latin letters by and to women (4th–13th century).

Only the `"Original letter:"` section is extracted; English translation, historical context, and scholarly apparatus are excluded. HTML tags within the Markdown are stripped.

```python
from latincyreaders import EpistolaeReader

reader = EpistolaeReader("/path/to/epistolae-hugo/content/letter")

# Latin text only
for text in reader.texts():
    print(text[:100])

# Letter metadata
for h in reader.headers():
    print(h["letter_id"], h["senders"], h["date"])
```

**Metadata per Doc:**

| Key | Example | Description |
|-----|---------|-------------|
| `letter_id` | `1` | Numeric letter ID from frontmatter |
| `title` | `Letter from Perpetua` | Letter title |
| `date` | `203` | Date from `ltr_date` field |
| `senders` | `["Perpetua"]` | List of sender names |
| `receivers` | `["Tertullian"]` | List of receiver names |
| `filename` | `1.html.md` | Source filename |

**License:** CC-BY-NC-SA 4.0.

### Epigraphic Database Heidelberg (EDHReader)

Read EpiDoc TEI-XML files from the [Epigraphic Database Heidelberg](https://github.com/epigraphic-database-heidelberg/data). The corpus contains ~82,000 Latin (and Greek) inscriptions from across the Roman Empire, encoded with full Leiden-convention markup.

Only files containing `<div type="edition" xml:lang="la">` are processed; Greek-only inscriptions are silently skipped. Abbreviations are expanded (`D(is)` → `Dis`), editor restorations are included, and erasures (`<del>`) are dropped.

```python
from latincyreaders import EDHReader

# Point at the cloned data repo
reader = EDHReader("/path/to/edh-data")

# Iterate Latin inscriptions as plain text
for text in reader.texts():
    print(text)

# Full NLP with line citations
for doc in reader.docs():
    meta = doc._.metadata
    print(f"{meta['hd_nr']} ({meta['not_before']}–{meta['not_after']} CE)")
    for line in doc.spans["lines"]:
        print(f"  {line._.citation}: {line.text}")

# Fast metadata scan (no NLP)
for h in reader.headers():
    print(h["hd_nr"], h["province"], h["type_of_inscription"])
```

**Metadata per Doc:**

| Key | Example | Description |
|-----|---------|-------------|
| `hd_nr` | `HD000001` | EDH inscription identifier |
| `not_before` | `71` | Earliest date CE (negative = BCE) |
| `not_after` | `130` | Latest date CE |
| `province` | `Latium et Campania (Regio I)` | Roman province |
| `type_of_inscription` | `epitaph` | Inscription type |
| `filename` | `HD000001.xml` | Source filename |

**Line spans** (`doc.spans["lines"]`): each Span carries `span._.citation` in the form `"HD000001.N"` (inscription ID + line number).

**Auto-download:** `EDHReader(auto_download=True)` will prompt to clone from GitHub on first use. The repo is ~500 MB; a `--depth 1` clone is used.

**License:** CC-BY-SA 4.0.

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

### Sentence Vector Search

Find semantically similar sentences across the corpus using sentence-level embeddings. Requires the `vectors` extra (`pip install latincyreaders[vectors]`).

```python
from latincyreaders import TesseraeReader
from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

reader = TesseraeReader()

# Build a vector index (saved to ~/latincy_data/vectors/<collection>/)
cfg = SentenceVectorConfig(collection="tesserae")
store = SentenceVectorStore(cfg)
store.build(reader)

# Semantic search
results = store.similar_to_sent("arma virumque cano", reader.nlp, top_k=5)
for r in results:
    print(f"[{r['score']:.3f}] {r['citation']}: {r['text'][:80]}")

# Or use the reader shortcut
results = reader.find_similar("amor", top_k=5, config=cfg)

# Auto-build on first query (builds index if none exists)
results = reader.find_similar("amor", auto_build=True)

# Find sentences similar to one already in the index
results = store.similar_to_doc_sent("vergil.aeneid.part.1.tess", 0, top_k=5)

# Index statistics
print(store.stats())
# {'collection': 'tesserae', 'sentences': 15800, 'vector_dim': 300, ...}
```

Vectors are stored as memory-mapped NumPy arrays for efficient search without external dependencies. See `notebooks/vector-search-demo.ipynb` for a full walkthrough.

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

### Persistent Disk Cache

For large corpora, enable persistent caching to avoid re-running the NLP pipeline across sessions. Cached documents are stored as `.spacy` DocBin files in `~/.latincy_cache/<collection>/` by default:

```python
from latincyreaders import TesseraeReader
from latincyreaders.cache.disk import CacheConfig

# Enable disk caching for the Tesserae corpus
config = CacheConfig(persist=True, collection="tesserae")
reader = TesseraeReader(model_name="la_core_web_lg", cache_config=config)

# First call runs NLP and caches to disk
doc = next(reader.docs(fileids="vergil.aeneid.part.1.tess"))

# Subsequent calls load from cache (~100x faster)
doc = next(reader.docs(fileids="vergil.aeneid.part.1.tess"))

# Custom cache location
config = CacheConfig(
    persist=True,
    collection="tesserae",
    cache_dir="/path/to/cache",
)

# Time-to-live (auto-expire after N days)
config = CacheConfig(persist=True, collection="tesserae", ttl_days=30)
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
- [digilibLT](http://digiliblt.uniupo.it) (Digital Library of Late-Antique Latin Texts)
- [Patristic Text Archive](https://pta.bbaw.de) (PTA — Greek and Latin patristic texts, CC-BY 4.0)
- [Universal Dependencies Latin Treebanks](https://universaldependencies.org/) (PROIEL, Perseus, ITTB, LLCT, UDante, CIRCSE)
- Any plaintext, TEI-XML, or CoNLL-U collection

## CLI Tools

Tools in `cli/`:

```bash
# Sentence search
python cli/reader_search.py --lemmas Caesar --limit 100
python cli/reader_search.py --forms Caesar Caesarem --limit 100
python cli/reader_search.py --pattern "\\bTheb\\w+" --output thebes.tsv

# Vector search — build and query sentence vector indices
python cli/vector_search.py build
python cli/vector_search.py build --collection vergil --fileids "vergil.*"
python cli/vector_search.py query "arma virumque cano" --top-k 10
python cli/vector_search.py stats
```

---

## Bibliography

- Bird, S., E. Loper, and E. Klein. 2009. *Natural Language Processing with Python*. O'Reilly: Sebastopol, CA.
- Bengfort, Benjamin, Rebecca Bilbro, and Tony Ojeda. 2018. *Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning*. O'Reilly: Sebastopol, CA.

---

*Developed by [Patrick J. Burns](http://github.com/diyclassics) with Claude Code in 2026.*
