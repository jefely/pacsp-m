"""Does 素参 exist? — a properly designed test.

WHY THIS SCRIPT EXISTS
----------------------
``g1_lebesgue_link.py`` reported the Lebesgue singular part as exactly zero on every corpus
and I presented that as a finding about 素参. It was not. It was a finding about the reference
measure I happened to pick.

    On a finite partition, the singular part is the mass on cells where nu = 0.
    I used nu = 1 (positive everywhere) and nu = distance-to-centroid (positive everywhere).
    A positive-everywhere nu forces dM = 0 BY CONSTRUCTION.

So that test could not have come out any other way, and the claim drawn from it is withdrawn.

WHAT 素参 ACTUALLY REQUIRES
---------------------------
For dM to be non-zero there must be a set S with nu(S) = 0 and Lambda(S) > 0: the individual's
trajectory must carry mass somewhere the substrate assigns NONE. Two consequences:

  * if nu is built from the same material as Lambda, nu > 0 everywhere Lambda goes, and
    素参 is identically zero — not because it is unmeasurable, but because it is empty;
  * 素参 can only be non-zero when nu is genuinely EXTERNAL — a public substrate that does
    not cover where this individual went.

THE TEST
--------
Take one arm as the individual and every OTHER arm pooled as the external public substrate.
For each of the individual's path points, measure the distance to the nearest substrate point.
A point is "covered" by the substrate at radius r if that distance is <= r; the substrate is
taken to assign zero mass beyond r, which is what makes the uncovered part singular rather
than merely rare.

Then sweep r and report the fraction of the path measure Lambda that is singular. The answer
is a CURVE, not a number, for the same reason ``math_variation_decompose`` returns a curve
over the truncation level: the split depends on where the cut is drawn, and reporting one
number would hide that.

Run:  python PACSP-M/exploration/g1_su_parameter_exists.py
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
sys.path.insert(0, str(M.parent / "toolkit" / "math"))

from g1_rn_density import ARMS, CACHE, delta_of  # noqa: E402

try:
    from pacsp_math.lebesgue import lebesgue_decompose
except Exception as exc:                                       # noqa: BLE001
    print(f"pacsp_math.lebesgue unavailable: {exc}", file=sys.stderr)
    raise SystemExit(2)


def unit(E: np.ndarray) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def main() -> int:
    print("素参是否存在？—— 一个设计正确的检验\n")
    print("=" * 78)
    print("先说清上一版错在哪")
    print("=" * 78)
    print("  g1_lebesgue_link.py 报「奇异部分恒为 0」，我把它当成关于素参的结论。")
    print("  **它不是。** 有限分割上奇异部分 = ν=0 的格子上的质量；")
    print("  而我用的 ν（计数测度、到质心距离）**处处为正**，")
    print("  所以 dM = 0 是构造上必然的，那个检验不可能给出别的结果。")
    print("  **该结论撤回。**\n")

    loaded = {}
    for arm in ARMS:
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if p.exists():
            loaded[arm] = unit(np.load(p))
    if not loaded:
        print("no cached embeddings", file=sys.stderr)
        return 2

    print("=" * 78)
    print("正确的检验：个体 vs 真正外来的公共基底")
    print("=" * 78)
    print("  个体 = 某一臂；公共基底 = 其余所有臂合起来（真正外来）")
    print("  覆盖半径 r：路径点到最近基底点的距离 <= r 算被覆盖")
    print("  基底在 r 之外记为**零质量**——这正是让未覆盖部分成为奇异而非仅仅稀疏的原因\n")

    results = {}
    print(f"  {'个体':<22}{'r/中位近邻距':>14}{'奇异占比':>11}{'被吸收':>10}")
    for arm, E in loaded.items():
        others = [e for a, e in loaded.items() if a != arm]
        if not others:
            continue
        sub = np.vstack(others)
        d = delta_of(E)
        pts = E[1:]                                   # each cell's leading endpoint
        # distance from each path point to the nearest substrate point
        dist = np.linalg.norm(pts[:, None, :] - sub[None, :, :], axis=2).min(axis=1)
        med = float(np.median(dist))
        curve = []
        for mult in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
            r = med * mult
            covered = (dist <= r).astype(float)
            # nu: positive ONLY on covered cells; zero beyond r
            nu = covered.copy()
            m = min(len(d), len(nu))
            if nu[:m].sum() == 0:
                frac_sing = 1.0
            else:
                res = lebesgue_decompose(mu=[float(x) for x in d[:m]],
                                         nu=[float(x) for x in nu[:m]])
                frac_sing = float(res.get("singular_fraction", float("nan")))
            curve.append({"r_over_median": mult, "singular_fraction": round(frac_sing, 4)})
            if mult in (0.0, 1.0, 3.0):
                print(f"  {arm:<22}{mult:>14.2f}{frac_sing:>11.4f}{1-frac_sing:>10.4f}")
        results[arm] = {"median_nn_dist": round(med, 4),
                        "curve": curve,
                        "singular_at_r_eq_median": next(c["singular_fraction"]
                                                        for c in curve
                                                        if c["r_over_median"] == 1.0)}

    print()
    print("=" * 78)
    print("读法")
    print("=" * 78)
    print("  r = 0            ：只承认与基底点**完全重合**的路径点被覆盖")
    print("  r = 中位近邻距   ：覆盖到一半的路径点（一个自然的基准）")
    print("  r 越大 → 奇异越少；这是必然的，因为覆盖在变大")
    print()
    print("  所以真正要问的不是「奇异是不是 0」，而是：")
    print("  **要把 r 放大到多少倍，奇异才降到可以忽略？**")
    print()
    for arm, r in results.items():
        c = r["curve"]
        # smallest multiplier at which the singular fraction drops below 5%
        small = next((x["r_over_median"] for x in c if x["singular_fraction"] < 0.05), None)
        tail = ("需要 r >= %.2f × 中位近邻距" % small) if small is not None else "在扫过的 r 内始终 >= 5%"
        print(f"  {arm:<22} 奇异降到 5% 以下：{tail}")

    print()
    print("  结论")
    print("    素参是否存在，**不是一个是/否问题，而是 ν 的选择问题**：")
    print("      · 若公共基底由**同一批材料**构成 → ν 处处为正 → 素参恒为 0，")
    print("        而且这不是「测不到」，是**空集**；")
    print("      · 只有当基底**真正外来**、覆盖不到个体去过的某些区域时，")
    print("        素参才非零——此时它度量的正是「公共基底无法表示的那一部分」。")
    print("    上表给出的，就是在「其余语料作为公共基底」这一具体选择下，")
    print("    素参随覆盖半径衰减的曲线。")

    out = M / "records_centroid" / "g1_su_parameter.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
