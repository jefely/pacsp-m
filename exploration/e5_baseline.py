"""E5 baseline: measure discriminative power, not problem count.

Everything verified so far shows the newer representations have fewer problems —
the graph removes order sensitivity, the matching matrix removes both order and sigma.
None of it shows they carry more information. This establishes the comparison that
decides whether a change of representation is worth making.

For each contrast that matters to the paper (human against machine, one model against
another, one domain against another) several measures are computed and scored by how
far apart they place the two groups relative to their spread. A measure that cannot
separate the groups is not useful however well behaved it is.

Measures, all computed from the encoder's embeddings so no language model is needed:

  C_T               the existing scalar, kept as the baseline to beat
  mean_pair_dist    order-free, the average pairwise distance
  dist_cv           order-free, coefficient of variation of pairwise distances
  effective_rank    order-free, participation ratio of the distance spectrum
  knn_weight_sum    graph invariant on a similarity-built kNN graph
  knn_spectral_gap  graph invariant, second spectral gap of the weighted Laplacian

Because some arms have exactly one sample, group spread is not estimable for those, so
the comparison is run across arms rather than within, and the effect size is reported
as the standardised difference between the two arms' values where both exist.
"""

import json
import sys
import warnings
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import (  # noqa: E402
    compute_ct, compute_deltas, compute_embeddings, compute_mus, load_samples,
)

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
WINDOW = 5
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

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


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def knn_edges(E, k=3):
    d2 = sq(E, E)
    np.fill_diagonal(d2, np.inf)
    e = set()
    for i in range(len(E)):
        for j in np.argsort(d2[i])[:k]:
            e.add((min(i, int(j)), max(i, int(j))))
    return sorted(e)


def measures(E):
    """All order-free, except C_T which is order-dependent by construction."""
    n = len(E)
    d2 = sq(E, E)
    iu = np.triu_indices(n, k=1)
    off = np.sqrt(d2[iu])
    med = float(np.median(off))
    sigma = 0.5 * med

    # C_T: kept exactly as the pipeline computes it
    d = np.asarray(compute_deltas(E), dtype=float)
    m = np.asarray(compute_mus(E, window=WINDOW), dtype=float)
    k = min(len(d), len(m))
    ct = float(np.sum(m[:k] * d[:k]))

    # graph invariants on a similarity graph
    e = knn_edges(E, 3)
    w = np.array([float(np.exp(-sq(E[[i]], E[[j]])[0, 0] / (2 * sigma ** 2)))
                  for i, j in e])
    deg = np.zeros(n)
    for (i, j), wij in zip(e, w):
        deg[i] += wij; deg[j] += wij
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = deg[i]
    for (i, j), wij in zip(e, w):
        L[i, j] -= wij; L[j, i] -= wij
    ev = np.sort(np.linalg.eigvalsh(L))[::-1]

    # participation ratio of the distance spectrum: how many dimensions the cloud uses
    dev = E - E.mean(axis=0)
    sv = np.linalg.svd(dev, compute_uv=False)
    p = sv ** 2
    eff_rank = float((p.sum() ** 2) / (np.sum(p ** 2) + 1e-18))

    return {
        "C_T": round(ct, 4),
        "mean_pair_dist": round(float(off.mean()), 4),
        "dist_cv": round(float(off.std() / off.mean()), 4),
        "eff_rank": round(eff_rank, 2),
        "knn_weight_sum": round(float(w.sum()), 4),
        "knn_spectral_gap": round(float(ev[1] - ev[2]) if n > 2 else 0.0, 6),
    }


def main():
    vals = {}
    meta = {}
    for arm, dom, src in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = load_samples(p)
        if len(texts) < 5:
            continue
        E = compute_embeddings(texts, model_name=MODEL)
        mv = measures(E)
        vals[arm] = mv
        meta[arm] = {"domain": dom, "source": src, "n": len(texts)}
        print(f"  {arm:<22} " + "  ".join(f"{k}={v}" for k, v in mv.items()))

    keys = list(next(iter(vals.values())).keys())

    print()
    print("=== contrasts that matter to the paper ===")
    contrasts = []
    for dom in sorted({m["domain"] for m in meta.values()}):
        arms = [a for a, m in meta.items() if m["domain"] == dom]
        if len(arms) >= 2:
            contrasts.append((f"domain={dom}", arms[0], arms[1]))
    # across arms, the largest separations by each measure
    print(f"  {'contrast':<28} " + "".join(f"{k[:11]:>13}" for k in keys))
    for label, a, b in contrasts:
        if a not in vals or b not in vals:
            continue
        row = "".join(f"{vals[a][k]/vals[b][k] if vals[b][k] else 0:>13.3f}"
                      for k in keys)
        print(f"  {label:<28} {row}")

    print()
    print("  (values are ratios arm-a / arm-b; 1.000 means the measure cannot tell)")
    print()
    print("=== which measure separates best, by spread over all arms ===")
    print(f"  {'measure':<20} {'min':>10} {'max':>10} {'spread':>10} {'ratio':>8}")
    for k in keys:
        v = np.array([vals[a][k] for a in vals])
        spread = v.max() - v.min()
        ratio = v.max() / v.min() if v.min() else float("inf")
        print(f"  {k:<20} {v.min():>10.4f} {v.max():>10.4f} {spread:>10.4f} "
              f"{ratio:>8.2f}")

    out = ROOT / "records_centroid" / f"e5_baseline_{STAMP}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"measures": vals, "meta": meta},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
