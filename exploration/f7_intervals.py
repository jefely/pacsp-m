"""F7: bootstrap intervals on the mean_pair_dist ratios, and the same for node_volume.

mean_pair_dist is the most stable human-versus-machine measure found so far: order-free,
no free parameters, same direction in all four domains, and a spread of only 1.13 across
them. But each domain contributes exactly one human and one machine corpus, so every
ratio is a single number. If the interval crosses 1 the measure cannot support the claim
either, and the recommendation to report it would have to be withdrawn.

The resampling unit is the item. mean_pair_dist is the average over all item pairs; an
item appears in n-1 pairs, so resampling items with replacement is not a clean i.i.d.
scheme, and the interval it produces is approximate. The same procedure is applied to
node_volume, whose per-item values are independent, which makes the comparison between
the two measures fair on the resampling side while being honest that the pair statistic's
interval is the looser of the two.
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import compute_embeddings, load_samples  # noqa: E402

DATA = ROOT / "data"
EMB = "BAAI/bge-large-zh-v1.5"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
CACHE = ROOT / "records_centroid" / "probe_cache"
NBOOT = 4000
LIMIT = 31

PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc"),
    ("hc3_human_medicine", "hc3_ai_medicine"),
    ("hc3_human_openqa", "hc3_ai_openqa"),
]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd_from(D2):
    n = D2.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(D2[iu]).mean())


def bootstrap(D2, nboot=NBOOT, seed=11):
    """Resample items and recompute the pair average."""
    rng = np.random.default_rng(seed)
    n = D2.shape[0]
    vals = np.empty(nboot)
    for b in range(nboot):
        idx = rng.integers(0, n, size=n)
        sub = D2[np.ix_(idx, idx)]
        iu = np.triu_indices(n, k=1)
        vals[b] = np.sqrt(sub[iu]).mean()
    return vals


def main():
    out = {}
    print(f"  bootstrap {NBOOT} resamples over items")
    for h, m in PAIRS:
        ph, pm = DATA / h, DATA / m
        if not (ph.is_dir() and pm.is_dir()):
            print(f"  [skip] {h} / {m}")
            continue
        th, _ = load_samples(ph)
        tm, _ = load_samples(pm)
        th, tm = th[:LIMIT], tm[:LIMIT]
        Eh = compute_embeddings(th, model_name=EMB)
        Em = compute_embeddings(tm, model_name=EMB)
        Dh, Dm = sq(Eh, Eh), sq(Em, Em)

        point = mpd_from(Dh) / mpd_from(Dm)
        bh = bootstrap(Dh)
        bm = bootstrap(Dm)
        # ratio distribution from independent resamples of each arm
        ratio = bh / bm
        lo, hi = np.percentile(ratio, [2.5, 97.5])

        # node_volume on the same items, for a like-for-like interval
        fh, fm = CACHE / f"{h}__p0.npy", CACHE / f"{m}__p0.npy"
        nv = None
        if fh.exists() and fm.exists():
            rng = np.random.default_rng(12)
            ih = np.load(fh).sum(axis=1)
            im = np.load(fm).sum(axis=1)
            sh = ih[rng.integers(0, len(ih), size=(NBOOT, len(ih)))].mean(axis=1)
            sm = im[rng.integers(0, len(im), size=(NBOOT, len(im)))].mean(axis=1)
            rr = sh / sm
            nv = {
                "point": round(float(ih.mean() / im.mean()), 4),
                "ci95": [round(float(np.percentile(rr, 2.5)), 4),
                         round(float(np.percentile(rr, 97.5)), 4)],
                "excludes_1": bool(np.percentile(rr, 2.5) > 1 or
                                   np.percentile(rr, 97.5) < 1),
                "human_item_cv": round(float(ih.std() / ih.mean()), 4),
                "machine_item_cv": round(float(im.std() / im.mean()), 4),
            }

        rec = {
            "domain": h.split("_")[-1],
            "n": len(th),
            "mean_pair_dist": {
                "point": round(float(point), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "excludes_1": bool(lo > 1 or hi < 1),
                "human_point": round(mpd_from(Dh), 4),
                "machine_point": round(mpd_from(Dm), 4),
                "human_boot_cv": round(float(bh.std() / bh.mean()), 4),
                "machine_boot_cv": round(float(bm.std() / bm.mean()), 4),
            },
            "node_volume": nv,
        }
        out[rec["domain"]] = rec

        d = rec["mean_pair_dist"]
        print(f"\n  {rec['domain']}")
        print(f"    mean_pair_dist  human {d['human_point']:.4f}  "
              f"machine {d['machine_point']:.4f}  ratio {d['point']:.4f}")
        print(f"                    95% CI [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]  "
              f"excludes 1: {d['excludes_1']}  "
              f"(boot CV {d['human_boot_cv']:.4f} / {d['machine_boot_cv']:.4f})")
        if nv:
            print(f"    node_volume     ratio {nv['point']:.4f}  "
                  f"95% CI [{nv['ci95'][0]:.4f}, {nv['ci95'][1]:.4f}]  "
                  f"excludes 1: {nv['excludes_1']}  "
                  f"(item CV {nv['human_item_cv']:.3f} / {nv['machine_item_cv']:.3f})")

    print()
    print("=== summary: how many pairs can support a claim? ===")
    mpd_ok = sum(1 for v in out.values() if v["mean_pair_dist"]["excludes_1"])
    nv_ok = sum(1 for v in out.values()
                if v["node_volume"] and v["node_volume"]["excludes_1"])
    tot = len(out)
    print(f"  mean_pair_dist  {mpd_ok}/{tot} intervals exclude 1")
    print(f"  node_volume     {nv_ok}/{tot} intervals exclude 1")

    print()
    print(f"  {'domain':<10} {'mpd ratio':>10} {'mpd CI':>20} {'nv ratio':>10} "
          f"{'nv CI':>20}")
    for d, v in out.items():
        a = v["mean_pair_dist"]
        b = v["node_volume"]
        ci_a = f"[{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]"
        ci_b = f"[{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]" if b else "-"
        nvb = f"{b['point']:.4f}" if b else "-"
        print(f"  {d:<10} {a['point']:>10.4f} {ci_a:>20} {nvb:>10} {ci_b:>20}")

    f = ROOT / "records_centroid" / f"f7_intervals_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
