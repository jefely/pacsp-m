"""N3: is the implementation faithful to the definition, and does localising mu fix order sensitivity?

The theory defines

    C_T = integral over the path of mu(t) dLambda(t)

with mu a path-local quantity. The implementation computes

    C_T = sum_k mu_k * delta_k

where mu_k = 1 - mean cosine similarity inside a window. Two questions follow, and both
are testable.

First, is mu_k path-local? The window makes it depend only on nearby embeddings, so it
is local in that sense, but the normalisation is over the window rather than the path.
The source conversation's own diagnosis was that mu_k is a global quantity used as a
local weight, which would make the implementation not a faithful discretisation.

Second, and more useful: if a more local mu removes the order sensitivity, then the
discrepancy is the defect and fixing it is the repair. If order sensitivity survives
localisation, then order dependence is intrinsic to summing increments over a sequence
and the metric has to be described as a sequence statistic rather than repaired.

Variants of mu compared:
    global     1 - mean cosine similarity over the whole corpus (the source's claim)
    window     the shipped implementation, radius 5
    narrow     radius 1, i.e. only immediate neighbours
    adjacent   1 - cos(v_{k-1}, v_{k+1}), strictly path-local
    none       mu = 1, which reduces C_T to total path length

For each, the order sensitivity is measured the same way as before: the as-filed value
against 200 random permutations of the same items.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
NPERM = 200
ARMS = ["poem", "machine_poem", "lyrics", "techdoc",
        "hc3_human_medicine", "hc3_ai_medicine"]


def mus_variant(E, kind, window=5):
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    N = E / (norms + 1e-8)
    cos = N @ N.T
    n = len(E)
    mus = []
    for k in range(n):
        if kind == "global":
            iu = np.triu_indices(n, k=1)
            mu = 1 - cos[iu].mean()
        elif kind == "window":
            s, e = max(0, k - window), min(n, k + window + 1)
            w = cos[s:e, s:e]
            iu = np.triu_indices_from(w, k=1)
            mu = 1 - w[iu].mean() if len(iu[0]) else 0.0
        elif kind == "narrow":
            s, e = max(0, k - 1), min(n, k + 2)
            w = cos[s:e, s:e]
            iu = np.triu_indices_from(w, k=1)
            mu = 1 - w[iu].mean() if len(iu[0]) else 0.0
        elif kind == "adjacent":
            a, b = max(0, k - 1), min(n - 1, k + 1)
            mu = 1 - cos[a, b] if a != b else 0.0
        elif kind == "none":
            mu = 1.0
        mus.append(float(mu))
    return mus


def ct_from(E, kind):
    d = np.asarray(pacsp_core.compute_deltas(E), dtype=float)
    m = np.asarray(mus_variant(E, kind), dtype=float)
    n = min(len(d), len(m))
    return float(np.sum(m[:n] * d[:n]))


def main():
    rng = np.random.default_rng(5)
    kinds = ["global", "window", "narrow", "adjacent", "none"]
    out = {}

    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, _ = pacsp_core.load_samples(p)
        E = pacsp_core.compute_embeddings(texts, model_name=MODEL)
        rec = {}
        print(f"\n=== {arm}  n={len(E)} ===")
        print(f"  {'mu variant':<12} {'C_T as-filed':>13} {'shuf mean':>11} "
              f"{'bias':>8} {'CV':>7} {'pct':>6}")
        for kind in kinds:
            asis = ct_from(E, kind)
            vals = []
            for _ in range(NPERM):
                idx = rng.permutation(len(E))
                vals.append(ct_from(E[idx], kind))
            vals = np.asarray(vals)
            bias = (asis - vals.mean()) / vals.mean() if vals.mean() else 0.0
            cv = vals.std() / vals.mean() if vals.mean() else 0.0
            pct = float((vals < asis).mean() * 100)
            rec[kind] = {"ct": round(asis, 4), "shuf_mean": round(float(vals.mean()), 4),
                         "bias": round(float(bias), 4), "cv": round(float(cv), 4),
                         "pct": round(pct, 1)}
            print(f"  {kind:<12} {asis:>13.4f} {vals.mean():>11.4f} "
                  f"{bias:>8.3f} {cv:>7.4f} {pct:>6.1f}")
        out[arm] = rec

    print()
    print("=== summary: does localising mu remove the order sensitivity? ===")
    print("  bias = (as-filed - shuffled mean) / shuffled mean; "
          "large |bias| means order decides the number")
    print()
    print(f"  {'arm':<22} " + "".join(f"{k:>11}" for k in kinds))
    for arm, rec in out.items():
        print(f"  {arm:<22} " + "".join(f"{rec[k]['bias']:>11.3f}" for k in kinds))
    print()
    print(f"  {'mean |bias|':<22} " + "".join(
        f"{np.mean([abs(out[a][k]['bias']) for a in out]):>11.3f}" for k in kinds))
    print(f"  {'mean CV':<22} " + "".join(
        f"{np.mean([out[a][k]['cv'] for a in out]):>11.4f}" for k in kinds))

    f = M / "records_centroid" / "n3_mu_localisation.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
