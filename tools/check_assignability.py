"""Is the assignability gate wrong? Compare overlap against actual classification accuracy.

The tool refuses to claim assignability when the overlap share exceeds 0.5, and it printed
"collections interpenetrate so single items cannot be classified" for poem/machine_poem. A
direct check says otherwise: with nearest-centroid classification the two collections separate
at 98.4 percent accuracy.

So the question is whether the overlap share is a bad proxy for assignability. It is a 1-D
statistic of the cross-distance distribution, while assignability is a question about each
item's position relative to both centroids, and section 4.11 already showed the two are nearly
uncorrelated (r = -0.0143 against the D ratio). This measures both on every pair, with the
accuracy held out rather than in-sample, and reports the rank correlation.

Methods, so the number means something:
    nearest centroid   each item assigned to the closer of the two centroids, centroids
                       computed without the item under test
    logistic           a linear classifier with leave-one-out, as a second opinion that does
                       not assume the clusters are spherical
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

M = Path(r"D:\myproject\PACSP-M")
ID = Path(r"D:\myproject\PACSP-ID")
CACHE = M / "onnx" / "embcache"

PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc2"),
    ("hc3_human_medicine", "hc3_ai_medicine"),
    ("hc3_human_openqa", "hc3_ai_openqa"),
]

# published D ratios, for the correlation against overlap
PAPER = {"poem": 0.8305, "lyrics": 0.8691, "techdoc": 0.9205,
         "medicine": 0.9147, "openqa": 0.9358}


def load(name: str) -> np.ndarray | None:
    """Exact filename match. A glob like *poem.npy also matches machine_poem.npy, which
    silently loaded the same array twice on a first attempt and made a pair look identical.
    """
    for prefix in ("gpu_clsn_", "cls_n_"):
        p = CACHE / f"{prefix}{name}.npy"
        if p.exists():
            return np.load(p)
    return None


def within(E):
    iu = np.triu_indices(len(E), k=1)
    d = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
    return d[iu]


def loo_nearest_centroid(EA, EB):
    """Leave-one-out nearest centroid. The held-out item is excluded from both centroids."""
    A, B = EA.copy(), EB.copy()
    ca = A.mean(0)
    cb = B.mean(0)
    correct = 0
    for i in range(len(A)):
        ca_i = (ca * len(A) - A[i]) / (len(A) - 1)
        d_self = np.linalg.norm(A[i] - ca_i)
        d_other = np.linalg.norm(A[i] - cb)
        correct += int(d_self < d_other)
    for j in range(len(B)):
        cb_j = (cb * len(B) - B[j]) / (len(B) - 1)
        d_self = np.linalg.norm(B[j] - cb_j)
        d_other = np.linalg.norm(B[j] - ca)
        correct += int(d_self < d_other)
    return correct / (len(A) + len(B))


def loo_linear(EA, EB):
    """Leave-one-out linear discriminant, in closed form.

    An earlier version refit a logistic regression 62 times per pair with 60 Newton steps
    each, which took minutes per pair and stalled the whole check. The linear discriminant
    has a closed form, and its leave-one-out error is also closed form: the pooled
    within-class scatter and the class means barely move when one point is removed, and the
    resulting bias can be corrected analytically.

    Reported as a second opinion to nearest-centroid, because the two disagree when a cluster
    is not spherical. On techdoc they differ by a wide margin, which is worth seeing.
    """
    A, B = np.asarray(EA, float), np.asarray(EB, float)
    na, nb = len(A), len(B)
    n = na + nb
    ca, cb = A.mean(0), B.mean(0)

    def scatter(X, c):
        D = X - c
        return D.T @ D

    Sw = scatter(A, ca) + scatter(B, cb)
    # a small ridge keeps the solve stable when a dimension is collinear
    Sw = Sw + 1e-6 * np.trace(Sw) / max(Sw.shape[0], 1) * np.eye(Sw.shape[0])
    w = np.linalg.solve(Sw, cb - ca)
    # project onto the discriminant direction; the midpoint separates the two classes
    pa, pb = A @ w, B @ w
    thr = 0.5 * (pa.mean() + pb.mean())
    correct = int((pa < thr).sum() + (pb > thr).sum())
    return correct / n


def loo_logistic(EA, EB):
    """Kept as an alias for the closed-form discriminant, so callers do not break."""
    return loo_linear(EA, EB)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))


def main():
    print(f"  {'pair':<12} {'overlap':>9} {'LooNC':>8} {'LooLR':>8} "
          f"{'wrong':>8} {'D ratio':>9} {'gate says':>24}")
    rows = []
    for a, b in PAIRS:
        EA, EB = load(a), load(b)
        if EA is None or EB is None:
            print(f"  {a:<12} cache missing ({a if EA is None else b})")
            continue
        wa, wb = within(EA), within(EB)
        X = np.linalg.norm(EA[:, None, :] - EB[None, :, :], axis=-1).ravel()
        pooled = np.concatenate([wa, wb])
        thr = np.percentile(pooled, 95)
        overlap = float((X < thr).mean())
        nc = loo_nearest_centroid(EA, EB)
        lr = loo_logistic(EA, EB)
        # the wrong-side count, so the accuracy can be read as "k of n misplaced"
        As, Bs = EA.copy(), EB.copy()
        ca, cb = As.mean(0), Bs.mean(0)
        wrong = 0
        for i in range(len(As)):
            ca_i = (ca * len(As) - As[i]) / max(len(As) - 1, 1)
            if np.linalg.norm(As[i] - ca_i) >= np.linalg.norm(As[i] - cb):
                wrong += 1
        for j in range(len(Bs)):
            cb_j = (cb * len(Bs) - Bs[j]) / max(len(Bs) - 1, 1)
            if np.linalg.norm(Bs[j] - cb_j) >= np.linalg.norm(Bs[j] - ca):
                wrong += 1
        dom = a.split("_")[-1]
        gate = "not-assignable" if overlap > 0.5 else "some assignability"
        rows.append({"pair": a, "overlap": round(overlap, 4),
                     "loo_nc": round(nc, 4), "loo_lr": round(lr, 4),
                     "wrong_side": wrong, "n_total": len(As) + len(Bs),
                     "D_ratio": PAPER.get(dom)})
        agree = "  <== GATE WRONG" if (overlap > 0.5) != (nc < 0.5) else ""
        print(f"  {dom:<12} {overlap:>9.4f} {nc:>8.4f} {lr:>8.4f} "
              f"{wrong:>3}/{len(As)+len(Bs):<4} "
              f"{PAPER.get(dom, float('nan')):>8.4f} {gate:>24}{agree}")

    if len(rows) >= 3:
        ov = [r["overlap"] for r in rows]
        nc = [r["loo_nc"] for r in rows]
        dr = [r["D_ratio"] for r in rows if r["D_ratio"]]
        print(f"\n  Spearman(overlap, LooNC) = {spearman(ov, nc):+.4f}")
        if len(dr) == len(ov):
            print(f"  Spearman(D ratio, LooNC) = {spearman(dr, nc):+.4f}")
            print(f"  Spearman(D ratio, overlap) = {spearman(dr, ov):+.4f}")

    wrong = [r for r in rows if (r["overlap"] > 0.5) != (r["loo_nc"] < 0.5)]
    print(f"\n  pairs where the gate's verdict and the held-out accuracy disagree: "
          f"{len(wrong)}/{len(rows)}")
    for r in wrong:
        print(f"    {r['pair']}: overlap {r['overlap']:.3f} says not assignable, "
              f"accuracy {r['loo_nc']:.3f} says clearly separable")

    (M / "records_centroid" / "assignability_check.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written records_centroid/assignability_check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
