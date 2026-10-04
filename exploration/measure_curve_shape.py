"""Measure the shape of the normalised deposition curve, and what it means.

The family fits say a 2-parameter Beta CDF describes seven of ten corpora with R^2
about 0.99 or better. But Beta(1,1) is the uniform distribution, whose CDF is exactly
the straight line y = x. So before treating Beta as a discovered law, this measures how
far each corpus's normalised curve actually departs from a straight line, and whether
the per-piece increments are uniform along the snapshot index.

If the increments are near-uniform, then the curve is essentially linear in snapshot
index and carries almost no information beyond its total: the shape is a consequence
of indexing by position, not a property of the work.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import (  # noqa: E402
    compute_deltas, compute_embeddings, compute_mus, load_samples,
)

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
WINDOW = 5

ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics", "techdoc",
        "machine_techdoc", "hc3_human_medicine", "hc3_ai_medicine",
        "hc3_human_openqa", "hc3_ai_openqa"]


def analyse(arm):
    p = DATA / arm
    if not p.is_dir():
        return None
    texts, _ = load_samples(p)
    if len(texts) < 5:
        return None
    emb = compute_embeddings(texts, model_name=MODEL)
    d = np.asarray(compute_deltas(emb), dtype=float)
    m = np.asarray(compute_mus(emb, window=WINDOW), dtype=float)
    n = min(len(d), len(m))
    inc = m[:n] * d[:n]
    cum = np.cumsum(inc)
    total = cum[-1]
    x = np.arange(1, n + 1) / n

    # 1. maximum absolute deviation of the normalised curve from the straight line
    norm_cum = cum / total
    dev = np.max(np.abs(norm_cum - x))

    # 2. coefficient of variation of the per-piece increments: 0 means perfectly even
    inc_cv = float(inc.std() / inc.mean()) if inc.mean() else None

    # 3. concavity: sign of the average of the second difference
    d2 = np.diff(inc)
    conc = "convex" if d2.mean() > 0 else ("concave" if d2.mean() < 0 else "linear")

    # 4. where the deposition concentrates: index of the half-mass point
    half = int(np.searchsorted(norm_cum, 0.5)) + 1
    half_frac = half / n

    # 5. how well a straight line through the origin fits
    r2_lin = 1 - float(np.sum((norm_cum - x) ** 2)) / float(
        np.sum((norm_cum - norm_cum.mean()) ** 2))

    return {
        "arm": arm, "T": n, "C_T": round(float(total), 4),
        "max_dev_from_line": round(float(dev), 4),
        "R2_linear": round(float(r2_lin), 4),
        "increment_CV": round(inc_cv, 4) if inc_cv is not None else None,
        "concavity": conc,
        "half_mass_at": half, "half_mass_frac": round(half_frac, 3),
    }


def main():
    rows = []
    for arm in ARMS:
        r = analyse(arm)
        if r:
            rows.append(r)

    print(f"  {'arm':<22} {'C_T':>8} {'maxdev':>8} {'R2lin':>7} "
          f"{'incCV':>7} {'shape':>8} {'half@':>6}")
    print("  " + "-" * 74)
    for r in rows:
        print(f"  {r['arm']:<22} {r['C_T']:>8.4f} {r['max_dev_from_line']:>8.4f} "
              f"{r['R2_linear']:>7.4f} {r['increment_CV']:>7.4f} "
              f"{r['concavity']:>8} {r['half_mass_frac']:>6.3f}")

    devs = [r["max_dev_from_line"] for r in rows]
    cvs = [r["increment_CV"] for r in rows]
    lin = [r["R2_linear"] for r in rows]
    print()
    print("=== summary ===")
    print(f"  R^2 of a straight line through the origin: "
          f"min {min(lin):.4f}  mean {np.mean(lin):.4f}")
    print(f"  max deviation from the line             : "
          f"mean {np.mean(devs):.4f}  max {max(devs):.4f}")
    print(f"  increment CV (0 = perfectly even)       : "
          f"min {min(cvs):.4f}  mean {np.mean(cvs):.4f}  max {max(cvs):.4f}")

    print()
    print("  Beta(1,1) is the uniform distribution and its CDF is the line y = x,")
    print("  so a Beta fit near a=1, b=1 is just a straight line being rediscovered.")
    betas = [r for r in rows]
    near_one = 0
    for r in rows:
        pass
    fits = json.loads((ROOT / "records_centroid" / "curve_fits.json").read_text(
        encoding="utf-8")) if (ROOT / "records_centroid" / "curve_fits.json").exists() else {}
    for arm, fams in fits.items():
        b = fams.get("beta CDF   I_x(a,b)", {}).get("norm", {}).get("params")
        if b and abs(b[0] - 1) < 0.25 and abs(b[1] - 1) < 0.25:
            near_one += 1
    print(f"  corpora whose Beta parameters sit within 0.25 of (1,1): "
          f"{near_one}/{len(fits)}")

    out = ROOT / "records_centroid" / "curve_shape.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
