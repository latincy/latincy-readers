#!/usr/bin/env python
"""CLI for building and querying Tesserae sentence vector indices.

Supports three subcommands:
- build: Index a corpus into a sentence vector store
- query: Semantic search by text
- stats: Show index info

Example usage:
    python vector_search.py build
    python vector_search.py build --collection vergil --fileids "vergil.*"
    python vector_search.py query "arma virumque cano"
    python vector_search.py query "de amicitia" --top-k 20
    python vector_search.py stats
"""

from __future__ import annotations

import argparse
import sys
import time

from latincyreaders import TesseraeReader
from latincyreaders.cache.vectors import SentenceVectorConfig, SentenceVectorStore


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and query Tesserae sentence vector indices.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index for full Tesserae corpus
  %(prog)s build

  # Build for specific author
  %(prog)s build --collection vergil --fileids "vergil.*"

  # Semantic search
  %(prog)s query "arma virumque cano"
  %(prog)s query "de amicitia" --top-k 20

  # Show index statistics
  %(prog)s stats
""",
    )

    parser.add_argument(
        "--collection",
        default="tesserae",
        help="Collection name for the vector store (default: tesserae)",
    )
    parser.add_argument(
        "--store-root",
        default=None,
        help="Root directory for vector stores (default: ~/latincy_data/vectors)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # build
    build_p = sub.add_parser("build", help="Build a sentence vector index")
    build_p.add_argument(
        "--fileids",
        default=None,
        help="Glob pattern to filter files (e.g. 'vergil.*')",
    )
    build_p.add_argument(
        "--corpus-root",
        default=None,
        help="Root directory of Tesserae corpus (default: auto-download)",
    )

    # query
    query_p = sub.add_parser("query", help="Search for similar sentences")
    query_p.add_argument("text", help="Query text")
    query_p.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results (default: 10)",
    )

    # stats
    sub.add_parser("stats", help="Show vector index statistics")

    return parser.parse_args(argv)


def _make_config(args: argparse.Namespace) -> SentenceVectorConfig:
    kwargs: dict = {"collection": args.collection}
    if args.store_root:
        kwargs["store_root"] = args.store_root
    return SentenceVectorConfig(**kwargs)


def cmd_build(args: argparse.Namespace) -> None:
    cfg = _make_config(args)

    print(f"Loading Tesserae corpus...")
    kwargs: dict = {}
    if args.corpus_root:
        kwargs["root"] = args.corpus_root
    reader = TesseraeReader(**kwargs)

    fileids = None
    if args.fileids:
        fileids = reader.fileids(match=args.fileids)
        if not fileids:
            print(f"No files matching '{args.fileids}'", file=sys.stderr)
            sys.exit(1)
        print(f"  Matched {len(fileids)} files")
    else:
        fileids = reader.fileids()
        print(f"  Found {len(fileids)} files")

    print(f"Building vector index (collection={cfg.collection!r})...")
    t0 = time.perf_counter()
    store = SentenceVectorStore(cfg)
    count = store.build(reader, fileids)
    elapsed = time.perf_counter() - t0

    stats = store.stats()
    print(f"  Indexed {count} sentences")
    print(f"  Vector dim: {stats['vector_dim']}")
    print(f"  Size: {stats['size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Store: {stats['store_dir']}")


def cmd_query(args: argparse.Namespace) -> None:
    cfg = _make_config(args)
    store = SentenceVectorStore(cfg)

    if store.stats()["sentences"] == 0:
        print(
            f"No index found for collection {cfg.collection!r}. "
            "Run 'build' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading NLP model for query vectorization...")
    from latincyreaders.nlp.pipeline import AnnotationLevel, get_nlp
    nlp = get_nlp(AnnotationLevel.FULL, model_name="la_core_web_lg", lang="la")

    print(f"Querying: {args.text!r}\n")
    results = store.similar_to_sent(args.text, nlp, top_k=args.top_k)

    for i, r in enumerate(results, 1):
        citation = r.get("citation", "")
        fileid = r["fileid"]
        score = r["score"]
        text = r["text"]

        header = citation if citation else fileid
        print(f"  {i:>3}. [{score:.3f}] {header}")
        print(f"       {text[:120]}")
        if len(text) > 120:
            print(f"       {text[120:240]}")
        print()


def cmd_stats(args: argparse.Namespace) -> None:
    cfg = _make_config(args)
    store = SentenceVectorStore(cfg)
    stats = store.stats()

    if stats["sentences"] == 0:
        print(f"No index found for collection {cfg.collection!r}.")
        return

    print(f"Collection: {stats['collection']}")
    print(f"Sentences:  {stats['sentences']:,}")
    print(f"Vector dim: {stats['vector_dim']}")
    print(f"Size:       {stats['size_bytes'] / 1024 / 1024:.1f} MB")
    print(f"Store dir:  {stats['store_dir']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    commands = {
        "build": cmd_build,
        "query": cmd_query,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
