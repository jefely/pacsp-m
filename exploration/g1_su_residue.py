"""素参 = 「非非非剩下的东西」——把「反复剥离后剩下的」做成可检验的东西。

WHY THIS IS NOT ANOTHER SINGLE-SUBSTRATE TEST
--------------------------------------------
Two earlier attempts are both withdrawn, and for opposite reasons:

  * ``g1_lebesgue_link.py`` used a nu with full support, so dM = 0 was forced by construction.
    A test that cannot fail proves nothing.
  * ``g1_su_parameter_exists.py`` used ONE external substrate and swept a coverage radius.
    Its critical multiplier came out at exactly 1.50x for all six arms, which is very nearly a
    definitional statement about the nearest-neighbour distance distribution — so the radius
    was doing the work, not the data.

Both shared a deeper flaw: they tried to pin 素参 down with ONE subtraction. But the
characterisation offered is "非非非剩下的东西" — what is left after repeated negation. A single
subtraction cannot produce a residue of *all* subtractions, so no single-substrate test can
speak to it. That is not a matter of picking a better nu.

WHAT IS ACTUALLY TESTABLE
-------------------------
A thing that resists definition cannot be measured, but the ELIMINATION PROCESS can be. So
this script does not try to measure 素参 at all. It measures:

    for a sequence of substrates S_1, S_2, ... that grow in expressive power,
    how much of the path measure Lambda survives each, and does the survivor converge?

The hypothesis under test is explicitly a convergence claim, and it can fail two ways:

    A. the residue is driven to 0 by external substrates          -> nothing survives; 素参 = 0
    B. the residue plateaus at a positive floor that is the SAME  -> a candidate residue exists
       across substrates built in completely different ways
    C. the plateau differs by construction family                 -> the "residue" is just
                                                                    whatever that family missed

Only B supports the existence of a residue worth the name. C is the outcome that would show
the whole notion is an artefact of the stripping procedure — and it is a real possibility,
which is why the experiment runs three unrelated families rather than one.

THE CONSTRAINT THAT MAKES IT MEANINGFUL
---------------------------------------
The substrates must be EXTERNAL: never allowed to include the individual's own points. If a
substrate may contain the individual, the residue is trivially zero (that is the degenerate
nu = Lambda case from G1) and the test is vacuous again.

Run:  python PACSP-M/exploration/g1_su_residue.py
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
from g1_rn_density import ARMS, CACHE, delta_of  # noqa: E402

EPS = [0.3, 0.5, 0.7]        # absolute chord distance on the unit sphere (range 0..2)
KS = [2, 4, 8, 16, 32, 64, 128]


def unit(E: np.ndarray) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def residue(pts: np.ndarray, d: np.ndarray, substrate: np.ndarray, eps: float) -> float:
    """Fraction of the path measure Lambda lying further than eps from the substrate.

    This is the stripping operator: everything within eps of the substrate is treated as
    absorbed by it, and whatever is left is what survived this round of negation.
    """
    if len(substrate) == 0:
        return 1.0
    dist = np.linalg.norm(pts[:, None, :] - substrate[None, :, :], axis=2).min(axis=1)
    m = min(len(dist), len(d))
    total = float(d[:m].sum())
    if total <= 0:
        return 0.0
    return float(d[:m][dist[:m] > eps].sum()) / total


def kmeans(X: np.ndarray, k: int, rng, iters: int = 30) -> np.ndarray:
    """Tiny k-means, no scipy. k-means++ style seeding by spread."""
    if k >= len(X):
        return X.copy()
    idx = rng.choice(len(X), size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2), axis=1)
        new = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else C[j]
                        for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return C


def main() -> int:
    rng = np.random.default_rng(7)
    loaded = {}
    for arm in ARMS:
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if p.exists():
            loaded[arm] = unit(np.load(p))

    print("素参 = 「非非非剩下的东西」—— 测剥离过程，不测素参本身\n")
    print("=" * 78)
    print("两个已撤回的检验，错法相反，但根子相同")
    print("=" * 78)
    print("  · g1_lebesgue_link.py ：ν 处处为正 → dM=0 是构造必然，检验不可能失败")
    print("  · g1_su_parameter_exists.py ：单个 ν + 扫描半径 → 六个语料临界倍数都是 1.50×，")
    print("                            那几乎是最近邻距离分布形状的定义性陈述，干活的是半径不是数据")
    print("  共同根子：**都想用「一次减法」把素参钉住。**")
    print("  而「非非非剩下的」是**全部减法之后**剩下的——一次减法产生不了它。")
    print("  这不是换一个更好的 ν 能解决的。\n")

    print("=" * 78)
    print("改为测：剥离序列的极限行为（三族互不相关的构造）")
    print("=" * 78)
    print("  约束：基底必须**外来**，绝不允许包含个体自己的点")
    print("        （否则残余平凡为零，又变成退化情形 ν=Λ）\n")

    family_results = {}
    for arm, E in loaded.items():
        others = {a: e for a, e in loaded.items() if a != arm}
        pooled = np.vstack(list(others.values()))
        d = delta_of(E)
        pts = E[1:]

        rec = {"genre": {}, "cluster": {}, "random": {}}

        # --- family A: add whole arms one at a time -------------------------
        acc, sizes = [], []
        for i, (a, e) in enumerate(others.items(), 1):
            acc.append(e)
            S = np.vstack(acc)
            sizes.append(len(S))
            rec["genre"][len(S)] = {f"{eps}": round(residue(pts, d, S, eps), 4)
                                    for eps in EPS}

        # --- family B: k-means centres fitted on the pooled substrate -------
        for k in KS:
            C = kmeans(pooled, k, rng)
            rec["cluster"][len(C)] = {f"{eps}": round(residue(pts, d, C, eps), 4)
                                      for eps in EPS}

        # --- family C: random draws from the pooled substrate ---------------
        for n in KS:
            n = min(n, len(pooled))
            S = pooled[rng.choice(len(pooled), size=n, replace=False)]
            rec["random"][len(S)] = {f"{eps}": round(residue(pts, d, S, eps), 4)
                                     for eps in EPS}

        family_results[arm] = rec

        print(f"  ── {arm}（个体 {len(E)} 点，外部基底池 {len(pooled)} 点）")
        print(f"     {'|S|':>6}{'族':>10}" + "".join(f"{'ε='+str(e):>10}" for e in EPS))
        for fam in ("genre", "cluster", "random"):
            for size in sorted(rec[fam]):
                vals = rec[fam][size]
                print(f"     {size:>6}{fam:>10}"
                      + "".join(f"{vals[str(e)]:>10.4f}" for e in EPS))
        print()

    # ---- the actual question: does a floor survive, and is it family-independent?
    print("=" * 78)
    print("判定：残余是被驱到 0，还是在三族之间收敛到同一个正的下界？")
    print("=" * 78)
    print(f"  {'个体':<22}{'ε':>6}{'genre 末端':>13}{'cluster 末端':>15}"
          f"{'random 末端':>14}{'判定':>22}")
    verdicts = {}
    for arm, rec in family_results.items():
        for eps in EPS:
            key = str(eps)
            def tail(fam):
                s = sorted(rec[fam])
                return rec[fam][s[-1]][key] if s else float("nan")
            g, c, r = tail("genre"), tail("cluster"), tail("random")
            ends = [g, c, r]
            if max(ends) < 0.02:
                v = "被驱到 0（无残余）"
            elif max(ends) - min(ends) < 0.10:
                v = f"三族同底 ≈{np.mean(ends):.3f} ← 候选"
            else:
                v = "三族不同底（程序伪影）"
            verdicts[f"{arm}|{eps}"] = {"genre": g, "cluster": c, "random": r, "verdict": v}
            print(f"  {arm:<22}{eps:>6.1f}{g:>13.4f}{c:>15.4f}{r:>14.4f}{v:>22}")

    cand = [k for k, v in verdicts.items() if "候选" in v["verdict"]]
    print()
    print("  ⚠ 对本次结果的撤回说明（务必先读这段）")
    print("    上表报出的「三族同底 18/18」**不构成证据**。")
    print("    探针（g1_su_residue_probe.py）查明：所有个体到外来基底的**中位最近距离在")
    print("    0.665 – 0.928** 之间，而本次用的 ε=0.3 与 ε=0.5 **低于全部六个个体**，")
    print("    所以那两档里基底**一个点都没吸收到**，残余 = 1.0000，三族当然一致。")
    print("    真正决定结果的是我选的 ε，不是剥离过程。")
    print("    只有 ε=0.7 落在有效区间（2/6 个体），样本不足以支持任何结论。")
    print()
    print("  怎么读（在撤回之后剩下的部分）")
    print("    · 这测的是**剥离过程**，不是素参的值——因为「不可定义」意味着没有值可测")
    print("    · 「被驱到 0」= 外来基底足以覆盖全部路径，没有东西剩下")
    print("    · 本次**没有**任何一个配置把残余驱到 0；但那是**没找到**，不是**证明了不存在**")
    print()
    print("  ⚠ 结构性上限（这条比上面的数值更重要）")
    print("    三族都在**同一个 1024 维嵌入空间**里取点。若素参被理解为")
    print("    「嵌入空间本身表达不了的东西」，那么**任何在这个空间里取基底的检验都够不着它**——")
    print("    这不是本次实验的不足，而是这类实验的**结构性上限**。")

    out = M / "records_centroid" / "g1_su_residue.json"
    out.write_text(json.dumps({"families": family_results, "verdicts": verdicts},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
