"""Does the stream-of-consciousness sedimentation measure separate human from agent?

The corpus measure D is order-invariant; S (S_flow / S_recur / S_sed) is order-aware
and is computed on the WITHIN-document stream. This script asks whether the within-
document sedimentation differs between the human arm and the machine arm of each
matched pair, and whether the true order carries sedimentation information at all
(permutation null model).

Reports, per pair:
    * unit counts (how many documents yield a real stream),
    * mean S_flow, S_recur, S_sed per arm,
    * the human/machine ratio with a bootstrap interval,
    * a permutation null: where the true S_sed sits within its own shuffled-order
      distribution (low percentile => the order deposits; middle => order-free).

The five pairs are the same as the paper's §4.3.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pacsp_core import compute_embeddings  # noqa: E402  (model loader kept identical)
from pacsp_stream import (  # noqa: E402
    document_sediment, permutation_sediment, segment_document,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
NPERM = 200
PERM_SEED = 11
PERM_SUBSET = 15          # permutation null on up to this many docs per arm
BOOT = 2000               # bootstrap resamples for the ratio interval
BOOT_SEED = 7

PAIRS = [
    ("poem", "machine_poem"),
    ("lyrics", "machine_lyrics"),
    ("techdoc", "machine_techdoc2"),
    ("hc3_human_medicine", "hc3_ai_medicine"),
    ("hc3_human_openqa", "hc3_ai_openqa"),
]


def arm_units(directory):
    """Return (per_file_unit_strings, files)."""
    units_by_file = []
    files = sorted(Path(directory).glob("*.txt"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        units_by_file.append((f.name, segment_document(text)))
    return units_by_file, files


def mean_and_boot_ratio(a, b, seed=BOOT_SEED, n=BOOT):
    """Bootstrap the ratio mean(S_sed[a]) / mean(S_sed[b])."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        if sb.mean() == 0:
            continue
        ratios.append(sa.mean() / sb.mean())
    ratios = np.asarray(ratios)
    return {
        "ratio": float(np.mean(a) / np.mean(b)),
        "ci_low": float(np.percentile(ratios, 2.5)),
        "ci_high": float(np.percentile(ratios, 97.5)),
    }


def main():
    report = {"model": MODEL, "pairs": []}

    for arm_a, arm_b in PAIRS:
        pair = {"arm_a": arm_a, "arm_b": arm_b}
        arms = {}
        for arm in (arm_a, arm_b):
            units_by_file, files = arm_units(DATA / arm)
            # embed every unit of the arm in one batch
            all_units = [u for _, us in units_by_file for u in us]
            per_file = {"files": len(files), "n_units": len(all_units),
                        "docs_with_stream": 0, "docs_too_short": 0,
                        "S_flow": [], "S_recur": [], "S_sed": [],
                        "perm_percentile": []}
            if not all_units:
                arms[arm] = per_file
                continue
            emb = compute_embeddings(all_units, model_name=MODEL)
            # slice per file
            idx = 0
            for fname, us in units_by_file:
                n = len(us)
                if n == 0:
                    continue
                e = emb[idx:idx + n]
                idx += n
                m = {"n": n, "S_flow": None, "S_recur": None, "S_sed": None}
                if n >= 2:
                    # local metrics without re-importing
                    steps = np.linalg.norm(e[1:] - e[:-1], axis=1)
                    sflow = float(steps.mean())
                    recur = np.empty(n - 1, dtype=float)
                    for k in range(1, n):
                        recur[k - 1] = np.linalg.norm(e[:k] - e[k], axis=1).min()
                    srecur = float(recur.mean())
                    m = {"n": n, "S_flow": sflow, "S_recur": srecur,
                         "S_sed": srecur / sflow if sflow > 0 else None}
                    per_file["docs_with_stream"] += 1
                    per_file["S_flow"].append(m["S_flow"])
                    per_file["S_recur"].append(m["S_recur"])
                    per_file["S_sed"].append(m["S_sed"])
                else:
                    per_file["docs_too_short"] += 1

                # permutation null on a subset
                if (n >= 3 and m["S_sed"] is not None
                        and len(per_file["perm_percentile"]) < PERM_SUBSET):
                    perms = np.asarray(permutation_sediment(e, NPERM, PERM_SEED))
                    if perms.size:
                        pct = float((perms <= m["S_sed"]).mean() * 100)
                        per_file["perm_percentile"].append(pct)

            arms[arm] = per_file

        a, b = arms[arm_a], arms[arm_b]
        pair["arm_a_summary"] = {
            "files": a["files"], "n_units": a["n_units"],
            "docs_with_stream": a["docs_with_stream"],
            "docs_too_short": a["docs_too_short"],
            "S_flow": round(float(np.mean(a["S_flow"])), 4) if a["S_flow"] else None,
            "S_recur": round(float(np.mean(a["S_recur"])), 4) if a["S_recur"] else None,
            "S_sed": round(float(np.mean(a["S_sed"])), 4) if a["S_sed"] else None,
            "perm_percentile_mean": round(float(np.mean(a["perm_percentile"])), 1)
            if a["perm_percentile"] else None,
        }
        pair["arm_b_summary"] = {
            "files": b["files"], "n_units": b["n_units"],
            "docs_with_stream": b["docs_with_stream"],
            "docs_too_short": b["docs_too_short"],
            "S_flow": round(float(np.mean(b["S_flow"])), 4) if b["S_flow"] else None,
            "S_recur": round(float(np.mean(b["S_recur"])), 4) if b["S_recur"] else None,
            "S_sed": round(float(np.mean(b["S_sed"])), 4) if b["S_sed"] else None,
            "perm_percentile_mean": round(float(np.mean(b["perm_percentile"])), 1)
            if b["perm_percentile"] else None,
        }
        if a["S_sed"] and b["S_sed"]:
            pair["ratio"] = mean_and_boot_ratio(
                np.asarray(a["S_sed"]), np.asarray(b["S_sed"]))
        else:
            pair["ratio"] = None
        report["pairs"].append(pair)

        # console table
        def line(tag, s):
            print(f"  {tag:<14} docs={s['docs_with_stream']:>2} "
                  f"units={s['n_units']:>4} S_flow={s['S_flow'] or float('nan'):>8.4f} "
                  f"S_recur={s['S_recur'] or float('nan'):>8.4f} "
                  f"S_sed={s['S_sed'] or float('nan'):>7.4f} "
                  f"perm%={s['perm_percentile_mean'] or float('nan'):>5.1f}")
        print(f"\n{arm_a}  vs  {arm_b}")
        line(arm_a, pair["arm_a_summary"])
        line(arm_b, pair["arm_b_summary"])
        if pair["ratio"]:
            r = pair["ratio"]
            print(f"  S_sed ratio (human/agent) = {r['ratio']:.4f}  "
                  f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    out = ROOT / "results" / "stream_sediment.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
