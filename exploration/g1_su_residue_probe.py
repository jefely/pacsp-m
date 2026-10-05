"""Sanity probe: is the free parameter (epsilon) doing the work, rather than the substrate?

``g1_su_residue.py`` reported "all three families reach the same floor" in 18/18 cases. If the
substrate barely changes the answer while epsilon changes it completely, that agreement is
vacuous — it says the threshold decides, not the stripping.

This measures the actual distance from each path point to the nearest EXTERNAL substrate
point, which is the quantity the threshold is being compared against.

Run:  python PACSP-M/exploration/g1_su_residue_probe.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(HERE))
from g1_rn_density import ARMS, CACHE  # noqa: E402


def unit(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    loaded = {a: unit(np.load(CACHE / f"gpu_clsn_{a}.npy"))
              for a in ARMS if (CACHE / f"gpu_clsn_{a}.npy").exists()}

    print("探针：干活的是阈值 ε，还是剥离过程？\n")
    print(f"  {'个体':<22}{'到外来基底的最近距离':>24}")
    print(f"  {'':<22}{'最小':>8}{'中位':>8}{'最大':>8}")
    rows = {}
    for arm, E in loaded.items():
        sub = np.vstack([e for a, e in loaded.items() if a != arm])
        pts = E[1:]
        d = np.linalg.norm(pts[:, None, :] - sub[None, :, :], axis=2).min(axis=1)
        rows[arm] = {"min": round(float(d.min()), 4), "median": round(float(np.median(d)), 4),
                     "max": round(float(d.max()), 4)}
        print(f"  {arm:<22}{d.min():>8.4f}{np.median(d):>8.4f}{d.max():>8.4f}")

    meds = [r["median"] for r in rows.values()]
    print()
    print(f"  所有个体的中位最近距离范围：{min(meds):.4f} – {max(meds):.4f}")
    print()
    print("  对照 g1_su_residue.py 用的三个阈值：0.3 / 0.5 / 0.7")
    print()
    for eps in (0.3, 0.5, 0.7):
        below = sum(1 for m in meds if m < eps)
        print(f"    ε = {eps}：{below}/{len(meds)} 个个体的中位最近距离小于它 "
              f"→ {'阈值有效' if below else '**阈值低于所有最近距离，基底完全不起作用**'}")
    print()
    print("  判定")
    if min(meds) > 0.5:
        print("    ε=0.3 与 ε=0.5 都**低于所有个体的典型最近距离**，")
        print("    所以那两行的「残余 = 1.0000、三族一致」是**平凡的**——")
        print("    基底根本没吸收到任何东西，三族当然一致。")
        print()
        print("    **结论：那次「三族同底」不构成证据。**")
        print("    真正决定结果的是我选的 ε，不是剥离过程。")
    else:
        print("    ε 落在有效区间内，三族一致性需要另外检验。")

    out = M / "records_centroid" / "g1_su_residue_probe.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
