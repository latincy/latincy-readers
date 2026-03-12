"""Demo: Build and query a Tesserae sentence vector index.

Shows end-to-end workflow:
1. Build a vector index from a subset of Tesserae files
2. Run semantic queries
3. Display results with citations and similarity scores

Run:
    .venv/bin/python cli/demo_vector_search.py
"""

import time
from pathlib import Path

from latincyreaders import TesseraeReader
from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore

# Use a temp directory so we don't pollute the user's data
STORE_ROOT = Path(__file__).parent / "cli_output" / "vectors"

# Subset of files for a quick demo
DEMO_FILES = [
    "vergil.aeneid.tess",
    "cicero.de_amicitia.tess",
    "caesar.de_bello_gallico.tess",
    "ovid.metamorphoses.tess",
]


def main():
    print("=" * 60)
    print("Tesserae Sentence Vector Search — Demo")
    print("=" * 60)

    # 1. Load reader
    print("\n1. Loading Tesserae corpus...")
    reader = TesseraeReader()
    all_fids = reader.fileids()
    print(f"   Corpus has {len(all_fids)} files total")

    # Find which demo files exist
    demo_fids = [f for f in DEMO_FILES if f in all_fids]
    if not demo_fids:
        # Fallback: use first 5 files
        demo_fids = all_fids[:5]
        print(f"   Demo files not found, using first {len(demo_fids)} files")
    else:
        print(f"   Using {len(demo_fids)} demo files: {demo_fids}")

    # 2. Build index
    cfg = SentenceVectorConfig(store_root=STORE_ROOT, collection="demo")
    store = SentenceVectorStore(cfg)

    print("\n2. Building sentence vector index...")
    t0 = time.perf_counter()
    count = store.build(reader, demo_fids)
    elapsed = time.perf_counter() - t0

    stats = store.stats()
    print(f"   Indexed {count:,} sentences in {elapsed:.1f}s")
    print(f"   Vector dim: {stats['vector_dim']}")
    print(f"   Index size: {stats['size_bytes'] / 1024:.0f} KB")

    # 3. Query
    queries = [
        "arma virumque cano",
        "amicitia",
        "Gallia est omnis divisa in partes tres",
        "amor",
    ]

    nlp = reader.nlp
    print("\n3. Semantic search results")
    print("-" * 60)

    for query in queries:
        print(f"\n   Query: {query!r}")
        results = store.similar_to_sent(query, nlp, top_k=5)

        for i, r in enumerate(results, 1):
            citation = r.get("citation", "") or r["fileid"]
            score = r["score"]
            text = r["text"][:100]
            print(f"   {i}. [{score:.3f}] {citation}")
            print(f"      {text}")

    # 4. Reader-level shortcut
    print("\n" + "-" * 60)
    print("\n4. Using reader.find_similar() shortcut:")
    results = reader.find_similar("ira deorum", top_k=3, config=cfg)
    for i, r in enumerate(results, 1):
        citation = r.get("citation", "") or r["fileid"]
        print(f"   {i}. [{r['score']:.3f}] {citation}: {r['text'][:80]}")

    # Cleanup note
    print(f"\n   Index stored at: {stats['store_dir']}")
    print("   (Run store.clear() or delete the directory to clean up)")


if __name__ == "__main__":
    main()
