"""Length control for the stream-of-consciousness sedimentation measure.

S_recur is the mean distance from each unit to its nearest EARLIER unit, so it shrinks
as a document grows (more prior units to be near). This confounds S_sed with document
length exactly as C_T was confounded with corpus length (§4.4 of the paper). The only
pair here whose documents are long enough to have a real stream — techdoc vs
machine_techdoc2 — also has a 7.3x length gap. This script truncates every document to
a common number of leading units K and re-measures, so the comparison is length-matched.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_core import compute_embeddings  # noqa: E402
from pacsp_stream import segment_document  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
KS = [5, 10, 15, 20, 30]
BOOT_SEED = 7


def metrics_for_prefix(emb, k):
    e = emb[:k]
    n = e.shape[0]
    if n < 2:
        return None
    steps = np.linalg.norm(e[1:] - e[:-1], axis=1)
    sflow = float(steps.mean())
    recur = np.empty(n - 1, dtype=float)
    for i in range(1, n):
        recur[i - 1] = np.linalg.norm(e[:i] - e[i], axis=1).min()
    srecur = float(recur.mean())
    return {"S_flow": sflow, "S_recur": srecur,
            "S_sed": srecur / sflow if sflow > 0 else None}


def load_arm(arm):
    out = []  # list of (n_units, embeddings)
    for f in sorted((DATA / arm).glob("*.txt")):
        units = segment_document(f.read_text(encoding="utf-8"))
        if len(units) < 2:
            continue
        emb = compute_embeddings(units, model_name=MODEL)
        out.append((len(units), emb))
    return out


def ratio_ci(a, b):
    rng = np.random.default_rng(BOOT_SEED)
    ratios = []
    for _ in range(2000):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        if sb.mean() == 0:
            continue
        ratios.append(sa.mean() / sb.mean())
    ratios = np.asarray(ratios)
    return (float(np.mean(a) / np.mean(b)),
            float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)))


def main():
    a = load_arm("techdoc")
    b = load_arm("machine_techdoc2")
    print(f"docs with >=2 units: techdoc {len(a)}, machine_techdoc2 {len(b)}")
    rows = []
    for k in KS:
        sa = [metrics_for_prefix(emb, k)["S_sed"]
              for n, emb in a if n >= k]
        sb = [metrics_for_prefix(emb, k)["S_sed"]
              for n, emb in b if n >= k]
        row = {
            "K": k,
            "n_a": len(sa), "n_b": len(sb),
            "S_sed_a": round(float(np.mean(sa)), 4),
            "S_sed_b": round(float(np.mean(sb)), 4),
        }
        if sa and sb:
            r, lo, hi = ratio_ci(np.asarray(sa), np.asarray(sb))
            row["ratio"] = round(r, 4)
            row["ci_low"] = round(lo, 4)
            row["ci_high"] = round(hi, 4)
        else:
            row["ratio"] = row["ci_low"] = row["ci_high"] = None
        rows.append(row)
        print(f"  K={k:>2}  n_a={row['n_a']:>2} n_b={row['n_b']:>2} "
              f"S_sed_a={row['S_sed_a']:.4f} S_sed_b={row['S_sed_b']:.4f} "
              f"ratio={row['ratio'] if row['ratio'] else float('nan'):.4f} "
              f"[{row['ci_low'] or float('nan'):.4f}, {row['ci_high'] or float('nan'):.4f}]")

    out = ROOT / "results" / "stream_length_control.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
