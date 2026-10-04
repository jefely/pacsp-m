"""Does C_T survive reordering the corpus?

C_T accumulates over adjacent snapshots, so it depends on the order the files happen
to be read in. For a body of work with no intrinsic sequence that order is arbitrary.
If shuffling the same texts changes C_T materially, then the corpus-level number is a
property of an accidental arrangement, not of the works.

This measures the spread of C_T over many random permutations, which also gives an
honest baseline for what "different" means when comparing two corpora.
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
    compute_ct, compute_deltas, compute_embeddings, compute_mus, load_samples,
)

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
WINDOW = 5
NPERM = 200

ARMS = ["poem", "machine_poem", "lyrics", "techdoc",
        "hc3_human_medicine", "hc3_ai_medicine"]


def main():
    rows = []
    for arm in ARMS:
        p = DATA / arm
        if not p.is_dir():
            continue
        texts, files = load_samples(p)
        if len(texts) < 5:
            continue
        # embed once; permutations only reorder the rows
        emb = compute_embeddings(texts, model_name=MODEL)
        rng = np.random.default_rng(11)
        vals = []
        for _ in range(NPERM):
            idx = rng.permutation(len(emb))
            e = emb[idx]
            d = np.asarray(compute_deltas(e), dtype=float)
            m = np.asarray(compute_mus(e, window=WINDOW), dtype=float)
            n = min(len(d), len(m))
            vals.append(float(np.sum(m[:n] * d[:n])))
        vals = np.asarray(vals)
        asis = float(np.sum(
            np.asarray(compute_mus(emb, window=WINDOW))[:len(compute_deltas(emb))] *
            np.asarray(compute_deltas(emb), dtype=float)[:len(compute_mus(emb, window=WINDOW))]))
        rows.append({
            "arm": arm,
            "C_T_as_ordered": round(asis, 4),
            "shuffle_mean": round(float(vals.mean()), 4),
            "shuffle_std": round(float(vals.std()), 4),
            "shuffle_min": round(float(vals.min()), 4),
            "shuffle_max": round(float(vals.max()), 4),
            "CV": round(float(vals.std() / vals.mean()), 4),
            "as_ordered_percentile": round(
                float((vals < asis).mean() * 100), 1),
        })

    print(f"  {'arm':<22} {'as-file':>8} {'shuf mean':>10} {'shuf sd':>8} "
          f"{'CV':>7} {'range':>18} {'pct':>6}")
    print("  " + "-" * 86)
    for r in rows:
        print(f"  {r['arm']:<22} {r['C_T_as_ordered']:>8.4f} "
              f"{r['shuffle_mean']:>10.4f} {r['shuffle_std']:>8.4f} "
              f"{r['CV']:>7.4f} "
              f"{'[' + format(r['shuffle_min'], '.2f') + ', ' + format(r['shuffle_max'], '.2f') + ']':>18} "
              f"{r['as_ordered_percentile']:>6.1f}")

    cvs = [r["CV"] for r in rows]
    print()
    print(f"  shuffle CV: min {min(cvs):.3f}  mean {np.mean(cvs):.3f}  "
          f"max {max(cvs):.3f}")
    print()
    print("  A CV near 0 means the number is a property of the works;")
    print("  a large CV means it is largely a property of the file order.")
    print("  The as-ordered column sits at percentile p of its own shuffle")
    print("  distribution, so p near 0 or 100 means the shipped order is unusual.")

    out = ROOT / "records_centroid" / "order_sensitivity.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
