"""Is C_T stable within a corpus (same source, same domain) or is it all noise?

The claim under test: "a given person in a given sub-domain has a similar C_T", i.e.
C_T behaves like a source-and-domain fingerprint rather than a discriminator between
human and machine.

Two things are needed to test it:
  1. a measure of how stable C_T is *within* one corpus
  2. a comparison against how much C_T moves *between* corpora

Stability is estimated by splitting an arm into halves and recomputing C_T on each,
repeated over every split point so ordering does not decide the answer, plus a
block-bootstrap interval over pair resamples. If within-corpus spread is comparable
to between-corpus differences, the arms are not separable and the claim is supported.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import (  # noqa: E402
    compute_ct, compute_deltas, compute_embeddings, compute_mus, load_samples,
)

DATA = ROOT / "data"
ARMS = [
    ("poem", "poem", "human-AI"),
    ("machine_poem", "poem", "autonomous"),
    ("lyrics", "lyrics", "human-AI"),
    ("machine_lyrics", "lyrics", "autonomous"),
    ("techdoc", "techdoc", "human-AI"),
    ("machine_techdoc", "techdoc", "autonomous"),
    ("hc3_human_medicine", "medicine", "human-only"),
    ("hc3_ai_medicine", "medicine", "AI-raw"),
    ("hc3_human_openqa", "openqa", "human-only"),
    ("hc3_ai_openqa", "openqa", "AI-raw"),
]

MODEL = "BAAI/bge-large-zh-v1.5"
WINDOW = 5
NBOOT = 400


def arm_stats(path):
    texts, files = load_samples(path)
    if len(texts) < 4:
        return None
    emb = compute_embeddings(texts, model_name=MODEL)
    d = np.asarray(compute_deltas(emb), dtype=float)
    m = np.asarray(compute_mus(emb, window=WINDOW), dtype=float)
    n = min(len(d), len(m))
    d, m = d[:n], m[:n]
    full = float(np.sum(m * d))

    # every half split, so the result does not depend on a single cut point
    halves = []
    for k in range(2, n - 1):
        halves.append(float(np.sum(m[:k] * d[:k])))
        halves.append(float(np.sum(m[k:] * d[k:])))
    halves = np.asarray(halves)

    rng = np.random.default_rng(7)
    block = max(2, WINDOW)
    nb = int(np.ceil(n / block))
    boot = []
    for _ in range(NBOOT):
        starts = rng.integers(0, max(1, n - block + 1), size=nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = idx[idx < n]
        boot.append(float(np.sum(m[idx] * d[idx])))
    boot = np.asarray(boot)

    sizes = [len(t.encode("utf-8")) for t in texts]
    return {
        "arm": path.name,
        "n": len(texts),
        "C_T": round(full, 4),
        "bytes_mean": int(np.mean(sizes)),
        "half_std": round(float(halves.std()), 4),
        "half_cv": round(float(halves.std() / halves.mean()), 4) if halves.mean() else None,
        "boo_lo": round(float(np.percentile(boot, 2.5)), 4),
        "boo_hi": round(float(np.percentile(boot, 97.5)), 4),
        "boo_cv": round(float(boot.std() / boot.mean()), 4) if boot.mean() else None,
    }


def main():
    t0 = time.time()
    rows = []
    for arm, domain, source in ARMS:
        p = DATA / arm
        if not p.is_dir():
            print(f"  [skip] {arm} missing")
            continue
        st = arm_stats(p)
        if not st:
            print(f"  [skip] {arm} too small")
            continue
        st["domain"] = domain
        st["source"] = source
        rows.append(st)
        print(f"  {arm:<22} C_T={st['C_T']:>8.4f}  half_cv={st['half_cv']:.3f}  "
              f"boot_cv={st['boo_cv']:.3f}  CI[{st['boo_lo']:.2f},{st['boo_hi']:.2f}]")

    print()
    print("=== within-corpus spread vs between-corpus differences ===")
    cvs = [r["half_cv"] for r in rows if r["half_cv"]]
    print(f"  half-split CV within arms: mean {np.mean(cvs):.3f} "
          f"max {max(cvs):.3f}")
    print("  (CV = std of the half-split C_T values divided by their mean)")
    print()
    print(f"  {'domain':<12} {'source':<12} {'C_T':>9} {'half_cv':>8}")
    for r in sorted(rows, key=lambda x: (x["domain"], x["C_T"])):
        print(f"  {r['domain']:<12} {r['source']:<12} {r['C_T']:>9.4f} "
              f"{r['half_cv']:>8.3f}")

    print()
    print("=== same domain, different source ===")
    for dom in sorted({r["domain"] for r in rows}):
        rs = [r for r in rows if r["domain"] == dom]
        if len(rs) < 2:
            continue
        lo = min(rs, key=lambda x: x["C_T"])
        hi = max(rs, key=lambda x: x["C_T"])
        ratio = hi["C_T"] / lo["C_T"]
        print(f"  {dom:<12} {lo['source']:<12} {lo['C_T']:>8.4f}   "
              f"{hi['source']:<12} {hi['C_T']:>8.4f}   ratio {ratio:.2f}")

    print()
    print("=== across domains, same source kind ===")
    for src in ("human-only", "AI-raw", "autonomous", "human-AI"):
        rs = [r for r in rows if r["source"] == src]
        if len(rs) < 2:
            continue
        lo = min(rs, key=lambda x: x["C_T"])
        hi = max(rs, key=lambda x: x["C_T"])
        print(f"  {src:<12} min {lo['C_T']:>8.4f} ({lo['domain']})   "
              f"max {hi['C_T']:>8.4f} ({hi['domain']})   ratio {hi['C_T']/lo['C_T']:.2f}")

    out = ROOT / "records_centroid" / "stability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
