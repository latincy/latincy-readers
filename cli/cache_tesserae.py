"""Cache Tesserae corpus through la_core_web_lg with DiskCache.

Usage:
    # Smoke test — one file:
    python cli/cache_tesserae.py --smoke

    # Full corpus:
    python cli/cache_tesserae.py
"""

from __future__ import annotations

import argparse

from tqdm import tqdm

from latincyreaders.cache.disk import CacheConfig
from latincyreaders.readers.tesserae import TesseraeReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache Tesserae via DiskCache")
    parser.add_argument("--smoke", action="store_true", help="Process one file only")
    args = parser.parse_args()

    config = CacheConfig(persist=True, collection="tesserae")
    reader = TesseraeReader(
        model_name="la_core_web_lg",
        cache_config=config,
    )

    fileids = reader.fileids()
    if args.smoke:
        fileids = fileids[:1]

    print(f"Caching {len(fileids)} file(s) to {config.cache_dir / 'tesserae'}")
    print(f"Model: la_core_web_lg\n")

    total_tokens = 0
    for fileid in tqdm(fileids, desc="Caching", unit="file"):
        doc = next(reader.docs(fileids=fileid))
        total_tokens += len(doc)
        tqdm.write(f"  {len(doc):>6} tokens  {fileid}")

    print(f"\nDone. {total_tokens:,} tokens cached across {len(fileids)} files.")


if __name__ == "__main__":
    main()
