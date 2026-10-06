<img src="https://raw.githubusercontent.com/latincy/latincy-readers/main/assets/latincy-readers-logo.jpg" alt="LatinCy Readers" width="400">

[![PyPI version](https://img.shields.io/badge/pypi-v1.9.0-orange.svg)](https://pypi.org/project/latincy-readers/)
[![Python versions](https://img.shields.io/pypi/pyversions/latincy-readers.svg)](https://pypi.org/project/latincy-readers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Corpus readers for Latin and Ancient Greek texts with LatinCy NLP integration.**

`latincy-readers` provides unified access to classical texts—from the Tesserae corpus to Perseus to the Patristic Text Archive and more—with automatic NLP processing via [LatinCy](https://github.com/diyclassics/latincy) spaCy models.

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
pip install https://huggingface.co/latincy/la_core_web_lg/resolve/main/la_core_web_lg-3.9.8-py3-none-any.whl

# Ancient Greek model (grc_dep_web_lg)
pip install https://huggingface.co/latincy/grc_dep_web_lg/resolve/main/grc_dep_web_lg-3.8.4-py3-none-any.whl
```

You can skip model installation if you only need raw text iteration (`AnnotationLevel.NONE`) or rule-based sentence splitting (`AnnotationLevel.MINIMAL`). All other levels require a model.

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
| `TesseraeReader` | `.tess` | Yes | LatinCy Latin Tesserae corpus |
| `GreekTesseraeReader` | `.tess` | Yes | LatinCy Greek Tesserae corpus |
| `PlaintextReader` | `.txt` | No | Plain text files |
| `LatinLibraryReader` | `.txt` | Yes | Latin Library corpus |
| `TEIReader` | `.xml` | No | TEI-XML documents |
| `PerseusReader` | `.xml` | No | Perseus Digital Library TEI |
| `CamenaReader` | `.xml` | Yes | CAMENA Neo-Latin corpus |
| `DigilibLTReader` | `.xml` | No | digilibLT Late-Antique Latin TEI corpus |
| `PTAReader` | `.xml` | Yes | Patristic Text Archive (Greek & Latin) |
| `CSELReader` | `.xml` | No | Corpus Scriptorum Ecclesiasticorum Latinorum |
| `EDHReader` | `.xml` | No | Epigraphic Database Heidelberg EpiDoc inscriptions |
| `FormulaeReader` | `.xml` | No | Formulae–Litterae–Chartae TEI charters |
| `EpistolaeReader` | `.html.md` | No | Epistolae medieval women's Latin letters |
| `ProjectGutenbergReader` | `.txt` | Yes (fetch) | Project Gutenberg plain-text files |
| `WikiSourceReader` | `.wiki` | Yes (fetch) | Latin Wikisource wikitext |
| `TxtdownReader` | `.txtd` | No | Txtdown format: citations, critical markup, speaker & cross-source quotation |
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

Read Ancient Greek texts from the LatinCy Greek Tesserae corpus using LatinCy Greek NLP models:

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

#### Model-version awareness

Cached annotations are stamped with the model that produced them (`model_name` +
`model_version`). The DocBin cache is treated as a **stamp-keyed ephemeral layer**:
if you upgrade the LatinCy model (say `la_core_web_lg` 3.9.4 → 3.9.6), the stamp no
longer matches, so the stale blob is discarded and the document is transparently
re-annotated with the new model. You never get sentence boundaries or tags from an
old model without knowing it.

The canonical `.conlluc` store (community-corrected, shipped with a corpus) is more
valuable and is **not** thrown away automatically. Instead, loading a `.conlluc`
built by a different model emits a warning so you can decide whether to rebuild:

```python
import warnings
from latincyreaders.warnings import AnnotationModelMismatchWarning

# ... reading docs whose canonical annotations predate your active model ...
# AnnotationModelMismatchWarning: canonical annotations for 'vergil.aen.tess'
#   were built with la_core_web_lg 3.9.4; active model is la_core_web_lg 3.9.6.
#   Boundaries/tags may be stale. Rebuild the canonical store (or delete the
#   .conlluc to regenerate) if you need annotations from the active model.

# Escalate to an error if a version mismatch should never pass silently:
warnings.simplefilter("error", AnnotationModelMismatchWarning)
```

Cache entries written before v1.8.0 carry no stamp and are rebuilt once on first
access — no manual cache deletion needed.

### Correcting Annotations

NLP output has errors. You can lock in corrections that make the cache trustworthy
and double as a training-ready record of human judgements. Corrections are a small
JSON overlay on the compact DocBin base cache, linked by durable opaque token ids:

```python
from latincyreaders import TesseraeReader, AnnotationLevel
from latincyreaders.cache.disk import CacheConfig

reader = TesseraeReader(
    annotation_level=AnnotationLevel.FULL,
    cache_config=CacheConfig(persist=True, collection="tesserae"),
)

doc = next(reader.docs("vergil.aeneid.part.1.tess"))
tok = doc[0]

# Lock in a lemma fix, keyed by the token's durable id.
reader.correct(
    "vergil.aeneid.part.1.tess", tok._.token_id,
    "lemma", "arma", evidence="clearly the noun here",
)

# Every later read overlays the gold value; the token is flagged.
doc = next(reader.docs("vergil.aeneid.part.1.tess"))
assert doc[0].lemma_ == "arma"
assert doc[0]._.corrected is True
```

Corrections (fields: `lemma`, `upos`, `xpos`, `feats`, `deprel`) are stored
**outside** the DocBin cache — in `~/latincy_data/corrections/<collection>/` by
default, or a directory you pass as `corrections_dir` — so they survive
`clear_cache()` and full re-annotation. The base DocBin is never mutated: it stays
"silver," and each correction records the machine value it overrode. When a model
upgrade or source edit re-tokenizes a text, corrections are re-pointed onto the new
tokenization (verified by surface form); anything that can't be placed safely is
quarantined to `corrections_unresolved.log` for manual review rather than
mis-applied.

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

- [Tesserae Latin Corpus](https://github.com/latincy/lat_text_tesserae)
- [Tesserae Greek Corpus](https://github.com/latincy/grc_text_tesserae)
- [Perseus Digital Library TEI](https://www.perseus.tufts.edu/)
- [Latin Library](https://github.com/cltk/lat_text_latin_library)
- [CAMENA Neo-Latin](https://github.com/nevenjovanovic/camena-neolatinlit)
- [digilibLT](http://digiliblt.uniupo.it) (Digital Library of Late-Antique Latin Texts)
- [Patristic Text Archive](https://pta.bbaw.de) (PTA — Greek and Latin patristic texts, CC-BY 4.0)
- [Corpus Scriptorum Ecclesiasticorum Latinorum](https://github.com/OpenGreekAndLatin/csel-dev) (Open Greek and Latin Project, CC-BY-SA 4.0)
- [Epigraphic Database Heidelberg](https://github.com/epigraphic-database-heidelberg/data) (EDH — EpiDoc TEI-XML, CC BY-SA 4.0)
- [Formulae–Litterae–Chartae](https://github.com/Formulae-Litterae-Chartae/formulae-open) (TEI-XML charters, CC BY 4.0)
- [Epistolae](https://github.com/ccnmtl/epistolae-hugo) (medieval women's Latin letters, Hugo Markdown, CC BY-NC-SA 4.0)
- [Project Gutenberg](https://www.gutenberg.org) (plain-text, fetched by ID)
- [Latin Wikisource](https://la.wikisource.org) (wikitext, fetched by page title)
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

### Method

- Bird, S., E. Loper, and E. Klein. 2009. *Natural Language Processing with Python*. O'Reilly: Sebastopol, CA.
- Bengfort, Benjamin, Rebecca Bilbro, and Tony Ojeda. 2018. *Applied Text Analysis with Python: Enabling Language-Aware Data Products with Machine Learning*. O'Reilly: Sebastopol, CA.

### Collections

- Tesserae Latin Corpus
  - Coffee, Neil, Jean-Pierre Koenig, Shakthi Poornima, Roelant Ossewaarde, Christopher Forstall, and Sarah Jacobson. 2012. "Intertextuality in the Digital Age." *Transactions of the American Philological Association* 142 (2): 383–422. https://doi.org/10.1353/apa.2012.0010.
  - Tesserae Project. 2026. *LatinCy Tesserae Latin Corpus*. Version 0.7.0. Edited by Patrick J. Burns and Classical Language Toolkit. https://github.com/latincy/lat_text_tesserae.
- Tesserae Greek Corpus
  - Tesserae Project. 2026. *LatinCy Tesserae Ancient Greek Corpus*. Version 0.7.2. Edited by Patrick J. Burns and Classical Language Toolkit. https://github.com/latincy/grc_text_tesserae.
- Perseus Digital Library TEI
  - Cerrato, Lisa, et al. 2026. *PerseusDL/canonical-latinLit*. Version 0.0.33195683883. Zenodo. https://doi.org/10.5281/zenodo.22149018.
- Latin Library
  - *The Latin Library*. https://www.thelatinlibrary.com. Via the CLTK repository: https://github.com/cltk/lat_text_latin_library.
- CAMENA Neo-Latin
  - *CAMENA: Corpus Automatum Multiplex Electorum Neolatinitatis Auctorum*. Directed by Wilhelm Kühlmann. Heidelberg and Mannheim, 1999–2013. GitHub archive by Neven Jovanović, 2016. https://github.com/nevenjovanovic/camena-neolatinlit.
  - Schibel, Wolfgang, and Jeffrey A. Rydberg-Cox. 2006. "Early Modern Culture in a Comprehensive Digital Library." *D-Lib Magazine* 12 (3). https://doi.org/10.1045/march2006-schibel.
- digilibLT
  - Borgna, Alice. 2017. "From Ancient Texts to Maps (and Back Again) in the Digital World. The DigilibLT Project." *Revista de Humanidades Digitales* 1: 296–313. https://doi.org/10.5944/rhd.vol.1.2017.16784.
  - Cattaneo, Gianmario, and Nadia Rosso. 2026. "DigilibLT: una biblioteca digitale." *Umanistica Digitale* 23: 23–30. https://doi.org/10.60923/issn.2532-8816/23594.
- Patristic Text Archive
  - von Stockhausen, Annette. 2024. *Patristisches Textarchiv. Ein Open Access-Archiv antiker christlicher Texte*. Version 1.1.12315518284. Zenodo. https://doi.org/10.5281/zenodo.14444959.
- Corpus Scriptorum Ecclesiasticorum Latinorum
  - Franzini, Greta, et al. 2022. *csel-dev*. Version 1.0.427. Zenodo. https://doi.org/10.5281/zenodo.6599925.
- Epigraphic Database Heidelberg
  - Grieshaber, Frank. 2019. "Epigraphic Database Heidelberg – Data Reuse Options." Heidelberg University Library. https://doi.org/10.11588/heidok.00026599.
- Formulae–Litterae–Chartae
  - Munson, Matthew. 2023. *formulae-open*. Zenodo. https://doi.org/10.5281/zenodo.10082666.
- Epistolae
  - Ferrante, Joan. 2014. *Epistolae: Medieval Women's Latin Letters*. Columbia University Libraries. https://doi.org/10.7916/RK1E-8X32.
- Project Gutenberg
  - *Project Gutenberg*. Project Gutenberg Literary Archive Foundation. https://www.gutenberg.org.
- Latin Wikisource
  - *Latin Wikisource*. Wikimedia Foundation. https://la.wikisource.org.
- Universal Dependencies Latin Treebanks
  - Nivre, Joakim, Marie-Catherine de Marneffe, Filip Ginter, Jan Hajič, Christopher D. Manning, Sampo Pyysalo, Sebastian Schuster, Francis Tyers, and Daniel Zeman. 2020. "Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection." In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, 4034–4043. Marseille: European Language Resources Association. https://aclanthology.org/2020.lrec-1.497/.
  - Gamba, Federica, and Daniel Zeman. 2023. "Universalising Latin Universal Dependencies: a harmonisation of Latin treebanks in UD." In *Proceedings of the Sixth Workshop on Universal Dependencies (UDW, GURT/SyntaxFest 2023)*, 7–16. Washington, D.C.: Association for Computational Linguistics. https://aclanthology.org/2023.udw-1.2/.
  - PROIEL
    - Haug, Dag T. T., and Marius L. Jøhndal. 2008. "Creating a Parallel Treebank of the Old Indo-European Bible Translations." In *Proceedings of the Second Workshop on Language Technology for Cultural Heritage Data (LaTeCH 2008)*, edited by Caroline Sporleder and Kiril Ribarov, 27–34.
  - Perseus
    - Bamman, David, and Gregory Crane. 2011. "The Ancient Greek and Latin Dependency Treebanks." In *Language Technology for Cultural Heritage*, edited by Caroline Sporleder, Antal van den Bosch, and Kalliopi Zervanou, 79–98. Berlin, Heidelberg: Springer. https://doi.org/10.1007/978-3-642-20227-8_5.
  - ITTB
    - Cecchini, Flavio Massimiliano, Marco Passarotti, Paola Marongiu, and Daniel Zeman. 2018. "Challenges in Converting the Index Thomisticus Treebank into Universal Dependencies." In *Proceedings of the Second Workshop on Universal Dependencies (UDW 2018)*, 27–36. Brussels. https://doi.org/10.18653/v1/W18-6004.
  - LLCT
    - Cecchini, Flavio Massimiliano, Timo Korkiakangas, and Marco Passarotti. 2020. "A New Latin Treebank for Universal Dependencies: Charters between Ancient Latin and Romance Languages." In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, 933–942. Marseille: European Language Resources Association. https://aclanthology.org/2020.lrec-1.117/.
  - UDante
    - Cecchini, Flavio Massimiliano, Rachele Sprugnoli, Giovanni Moretti, and Marco Passarotti. 2020. "UDante: First Steps Towards the Universal Dependencies Treebank of Dante's Latin Works." In *Proceedings of the Seventh Italian Conference on Computational Linguistics*, edited by Johanna Monti, Felice Dell'Orletta, and Fabio Tamburini. CEUR Workshop Proceedings 2769. https://ceur-ws.org/Vol-2769/paper_14.pdf.
  - CIRCSE
    - *UD_Latin-CIRCSE*. CIRCSE Research Centre, Milan. https://github.com/UniversalDependencies/UD_Latin-CIRCSE.

---

*Developed by [Patrick J. Burns](http://github.com/diyclassics) with Claude Code in 2026.*
