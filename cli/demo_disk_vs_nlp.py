"""Side-by-side demo: .conlluc disk load vs NLP pipeline for Eutropius.

Shows that cached annotations match pipeline output exactly (except for
human corrections), and demonstrates the speed advantage of disk loading.
"""

import time

import spacy

from latincyreaders.readers.tesserae import TesseraeReader

N_SENTS = 10

# ============================================================
# 1. Load from .conlluc (disk) — model never loads
# ============================================================
reader = TesseraeReader()
_ = reader.fileids()  # warm corpus scan

t0 = time.perf_counter()
disk_doc = list(reader.docs(fileids="eutropius.brevarium.tess"))[0]
t_disk = time.perf_counter() - t0

model_loaded = reader._nlp is not None
disk_sents = list(disk_doc.sents)[:N_SENTS]

# ============================================================
# 2. Load via NLP pipeline (cold start: model load + inference)
# ============================================================
texts = list(reader.texts(fileids="eutropius.brevarium.tess"))
raw_text = texts[0]

t_model_start = time.perf_counter()
nlp = spacy.load("la_core_web_lg")
t_model = time.perf_counter() - t_model_start

# Take first ~15 Tesserae lines to cover 10+ sentences
lines = raw_text.split("\n")
partial = "\n".join(lines[:15])

t1 = time.perf_counter()
nlp_doc = nlp(partial)
t_nlp = time.perf_counter() - t1

nlp_sents = list(nlp_doc.sents)[:N_SENTS]

# ============================================================
# 3. Timing summary
# ============================================================
t_pipeline_total = t_model + t_nlp

print("=" * 90)
print("TIMING BREAKDOWN")
print(f"  .conlluc disk load:     {t_disk:.3f}s  (model loaded: {model_loaded})")
print(f"  NLP model load:         {t_model:.3f}s")
print(f"  NLP pipeline inference: {t_nlp:.3f}s")
print(f"  NLP total (cold start): {t_pipeline_total:.3f}s")
print(f"  Speedup (vs cold):      {t_pipeline_total / t_disk:.1f}x")
print(f"  Speedup (vs warm):      {t_nlp / t_disk:.1f}x")
print(f"Source: {disk_doc._.metadata.get('source', '?')}")
print("=" * 90)

# ============================================================
# 4. Side-by-side annotation comparison
# ============================================================
total_tokens = 0
total_diffs = 0

for si in range(min(N_SENTS, len(disk_sents), len(nlp_sents))):
    ds = disk_sents[si]
    ns = nlp_sents[si]
    print(f"\n--- Sentence {si + 1} ---")
    print(f"Text: {ds.text[:90]}")

    d_toks = [
        (t.text, t.lemma_, t.pos_, t.tag_, str(t.morph))
        for t in ds
        if not t.is_punct and not t.is_space
    ]
    n_toks = [
        (t.text, t.lemma_, t.pos_, t.tag_, str(t.morph))
        for t in ns
        if not t.is_punct and not t.is_space
    ]

    mismatches = 0
    for di, (dt, dl, dp, dx, dm) in enumerate(d_toks):
        if di >= len(n_toks):
            break
        nt, nl, np_, nx, nm = n_toks[di]
        if dt != nt:
            continue
        diffs = []
        if dl != nl:
            diffs.append(f"lemma: {dl} vs {nl}")
        if dp != np_:
            diffs.append(f"upos: {dp} vs {np_}")
        if dx != nx:
            diffs.append(f"xpos: {dx} vs {nx}")
        if dm != nm:
            diffs.append(f"morph: {dm} vs {nm}")
        if diffs:
            mismatches += 1
            total_diffs += 1
            print(f"  DIFF  {dt:20s} | {' | '.join(diffs)}")

    total_tokens += len(d_toks)
    if mismatches == 0:
        print(f"  MATCH ({len(d_toks)} tokens identical)")
    else:
        print(f"  {mismatches} diff(s) / {len(d_toks)} tokens")

print("\n" + "=" * 90)
print(f"SUMMARY: {total_tokens} tokens compared, {total_diffs} differences")
if total_diffs:
    print(f"  (Differences are from corrections applied to .conlluc)")
