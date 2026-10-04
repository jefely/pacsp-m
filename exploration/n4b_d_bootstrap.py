"""N4b: is D's 5-of-5 direction real, or is the effect too small to reverse?

Route D gave ratios of 0.9883 to 0.9991, a spread of 1.01. Five of five below 1 looks
like perfect consistency, but a measure whose effect is one percent can appear consistent
simply because noise is not large enough to flip it. The question is whether that one
percent is distinguishable from zero.

This bootstraps D at the item level for each pair, in the same way mean_pair_dist was
bootstrapped, so the two can be compared directly. It also checks what D's spread of 1.01
means for cross-corpus comparison in practice.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
NBOOT = 4000
PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", "machine_techdoc"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]

from n4_sigma_resolutions import barycentre, median_dist, profile, shape_stats, sq


def d_stat(E):
    """Route D's scalar: the standardised shape statistics of the mean profile."""
    s0 = 0.5 * median_dist(E)
    s = shape_stats(profile(E, s0, "gaussian"))
    return float(np.linalg.norm(s / (np.abs(s).sum() + 1e-300)))


def main():
    embs = {}
    for h, m in PAIRS:
        for arm in (h, m):
            p = DATA / arm
            if p.is_dir() and arm not in embs:
                texts, _ = pacsp_core.load_samples(p)
                embs[arm] = pacsp_core.compute_embeddings(texts, model_name=MODEL)

    print("=== D: point estimate and bootstrap interval per pair ===")
    print(f"  {'domain':<10} {'D human':>10} {'D machine':>11} {'ratio':>9} "
          f"{'95% CI':>20} {'excl 1':>7}")
    rng = np.random.default_rng(13)
    out = {}
    for h, m in PAIRS:
        if h not in embs or m not in embs:
            continue
        # item-level bootstrap: recompute the corpus statistic on resampled items
        bh, bm = [], []
        for arm, store in ((h, bh), (m, bm)):
            E = embs[arm]
            n = len(E)
            for _ in range(NBOOT):
                idx = rng.integers(0, n, size=n)
                store.append(d_stat(E[idx]))
        bh, bm = np.asarray(bh), np.asarray(bm)
        ph, pm = d_stat(embs[h]), d_stat(embs[m])
        ratio = bh / bm
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        key = h.split("_")[-1]
        out[key] = {"d_human": round(ph, 6), "d_machine": round(pm, 6),
                    "ratio": round(ph / pm, 6),
                    "ci95": [round(float(lo), 6), round(float(hi), 6)],
                    "excludes_1": bool(lo > 1 or hi < 1),
                    "boot_cv_human": round(float(bh.std() / bh.mean()), 6),
                    "boot_cv_machine": round(float(bm.std() / bm.mean()), 6)}
        r = out[key]
        print(f"  {key:<10} {ph:>10.6f} {pm:>11.6f} {ph/pm:>9.6f} "
              f"{'[' + format(lo, '.6f') + ', ' + format(hi, '.6f') + ']':>20} "
              f"{str(r['excludes_1']):>7}")

    print()
    print("=== what a spread of 1.01 means ===")
    pts = [v["ratio"] for v in out.values()]
    print(f"  D ratios        : {min(pts):.6f} - {max(pts):.6f}  spread {max(pts)/min(pts):.4f}")
    f7 = M / "results"
    ref = None
    for p in f7.glob("f7_intervals*.json"):
        ref = json.loads(p.read_text(encoding="utf-8"))
    if ref:
        rp = [ref[k]["mean_pair_dist"]["point"] for k in ref]
        print(f"  mean_pair_dist  : {min(rp):.4f} - {max(rp):.4f}  spread {max(rp)/min(rp):.4f}")

    n_excl = sum(1 for v in out.values() if v["excludes_1"])
    print(f"\n  intervals excluding 1: {n_excl}/{len(out)}")
    cvs = [v[k] for v in out.values() for k in ("boot_cv_human", "boot_cv_machine")]
    print(f"  bootstrap CV range   : {min(cvs):.6f} - {max(cvs):.6f}")

    print()
    print("=== interpretation ===")
    if n_excl == len(out):
        print("  every pair separates, and the effect while small is not zero")
    else:
        print(f"  {len(out) - n_excl} pair(s) do NOT separate; the 5-of-5 direction count")
        print("  reflects a small effect rather than a robust one")
    if max(cvs) < 0.05:
        print("  the per-item variation is small enough that a one-percent effect is")
        print("  measurable, which is a genuine strength of this route")
    else:
        print("  the per-item variation is large relative to the effect")

    f = M / "records_centroid" / "n4b_d_bootstrap.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
