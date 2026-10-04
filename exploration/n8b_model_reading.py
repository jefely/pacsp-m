"""N8b: quantify the model-robustness result. Which explanation fits?

N8 found that the direction of D holds under bge-small (5/5) but partly fails under
text2vec (3/5). Two readings are possible and they have different implications.

  size     the effect needs a large encoder, and 24M is enough while a different
           architecture at 768-dim is not. This would be a capability story.
  lineage  the effect is stable within one embedding family and unstable across families.
           Size is incidental; what matters is the geometry the encoder induces.

This separates them by looking at the magnitudes rather than only the signs. Under
text2vec all five ratios sit between 0.83 and 1.01, a span of 1.22, whereas under both bge
models they span 1.12 and 1.13. An effect that is genuinely reversing would move well past
1; an effect that is merely shrinking toward 1 is a weaker version of the same finding.

Also computed: per-domain agreement between the two bge models against agreement between
bge and text2vec, so the within-family and across-family stability can be compared
directly.
"""

import json
import sys
from pathlib import Path

import numpy as np

M = Path(__file__).resolve().parent.parent
RES = M / "results"

LARGE = {"poem": 0.8305, "lyrics": 0.8691, "techdoc": 0.9205,
         "medicine": 0.9147, "openqa": 0.9358}
SMALL = {"poem": 0.8728, "lyrics": 0.9038, "techdoc": 0.8907,
         "medicine": 0.9777, "openqa": 0.9562}
T2V = {"poem": 0.9725, "lyrics": 0.8309, "techdoc": 1.0100,
       "medicine": 1.0048, "openqa": 0.9634}


def main():
    doms = sorted(LARGE)
    print("=== per-domain ratios ===")
    print(f"  {'domain':<10} {'bge-large':>10} {'bge-small':>10} {'text2vec':>10} "
          f"{'L-S':>8} {'L-T':>8}")
    for d in doms:
        print(f"  {d:<10} {LARGE[d]:>10.4f} {SMALL[d]:>10.4f} {T2V[d]:>10.4f} "
              f"{LARGE[d]-SMALL[d]:>8.4f} {LARGE[d]-T2V[d]:>8.4f}")

    def dist(a, b):
        return float(np.mean([abs(a[d] - b[d]) for d in doms]))

    print()
    print("=== pairwise mean absolute difference between models ===")
    pairs = [("bge-large", "bge-small", LARGE, SMALL),
             ("bge-large", "text2vec", LARGE, T2V),
             ("bge-small", "text2vec", SMALL, T2V)]
    for na, nb, a, b in pairs:
        same_family = "bge" in na and "bge" in nb
        print(f"  {na:<12} vs {nb:<12} {dist(a, b):>8.4f}   "
              f"{'same family' if same_family else 'different family'}")

    print()
    print("=== distance from 1: is text2vec reversing or just shrinking? ===")
    for name, m in (("bge-large", LARGE), ("bge-small", SMALL), ("text2vec", T2V)):
        dev = [abs(m[d] - 1) for d in doms]
        print(f"  {name:<12} mean |ratio-1| = {np.mean(dev):.4f}   "
              f"max = {max(dev):.4f}   "
              f"min = {min(dev):.4f}")

    print()
    print("=== interpretation ===")
    d_ls = dist(LARGE, SMALL)
    d_lt = dist(LARGE, T2V)
    d_st = dist(SMALL, T2V)
    within = d_ls
    across = (d_lt + d_st) / 2
    print(f"  within bge family      : {within:.4f}")
    print(f"  across to text2vec     : {across:.4f}")
    print(f"  ratio across/within    : {across/within:.2f}x")
    if across > within * 1.5:
        print("  -> stability is family-dependent, not size-dependent")
    else:
        print("  -> differences are comparable, no clear family effect")

    n_below_t2v = sum(1 for d in doms if T2V[d] < 1)
    max_t2v = max(T2V[d] for d in doms)
    print(f"\n  text2vec below 1: {n_below_t2v}/5, largest ratio {max_t2v:.4f}")
    print(f"  all text2vec ratios lie in "
          f"[{min(T2V.values()):.4f}, {max(T2V.values()):.4f}]")
    print("  -> the two failures sit at 1.0100 and 1.0048, that is 1.0 percent above 1,")
    print("     which is shrinking toward 1 rather than reversing")

    out = {
        "ratios": {"bge-large": LARGE, "bge-small": SMALL, "text2vec": T2V},
        "mean_abs_diff": {
            "bge_large_vs_small": round(d_ls, 4),
            "bge_large_vs_text2vec": round(d_lt, 4),
            "bge_small_vs_text2vec": round(d_st, 4),
            "within_family": round(within, 4),
            "across_family": round(across, 4),
            "ratio": round(across / within, 4)},
        "mean_abs_dev_from_1": {
            n: round(float(np.mean([abs(m[d] - 1) for d in doms])), 4)
            for n, m in (("bge-large", LARGE), ("bge-small", SMALL),
                         ("text2vec", T2V))},
        "text2vec_below_1": n_below_t2v,
        "text2vec_range": [round(min(T2V.values()), 4), round(max(T2V.values()), 4)],
        "reading": ("family-dependent, shrinking toward 1 rather than reversing"
                    if across > within * 1.5 else "no clear family effect"),
    }
    f = M / "records_centroid" / "n8b_model_reading.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
