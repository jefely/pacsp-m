"""命题 F，认真的检验：有限的外部基底能否把残余驱到 0？

三次失败的原因（写在 SU-PARAMETER.md §1）：每一次，决定结果的都是实验者引入的自由参数。
所以这次的设计要求是硬的：

    ① 基底足够丰沛   —— 用全部可得的文档，不是 5 个语料
    ② 基底真正外部   —— 个体所在的那一臂整体排除
    ③ **无自由参数** —— 没有阈值，没有半径；尺度由数据自己定，零假设是经验分布自身

三个自由度是怎么消掉的
----------------------
  · 尺度：不再选 ε，而是用池自身的典型距离 D̄（随机文档对的中位距离）做单位
  · 判据：不再选"多小算被吸收"，而是看 s(d) = r(d)/D̄ 的**分布本身**
  · 零假设：不再假设一个分布，而是用**池中所有文档的经验分布**当零假设——
            每个文档都被同样对待，所以离群与否由它自己所在的分布决定

统计量
------
    r(d) = 从文档 d 到【不属于 d 所在臂】的最近文档的距离
    s(d) = r(d) / D̄         D̄ = 池中随机文档对的中位距离

    s(d) << 1  →  d 有很近的外部邻居，被覆盖
    s(d) ≈ 1   →  d 的最近外部邻居和一个随机文档一样远 → **没有外部之物够得到它**
    s(d) >> 1  →  d 比随机还孤立

F 的判据
--------
    逐步扩大基底（逐个加入别的臂），看 s 的分布：
      上尾持续向 0 收缩  →  残余被驱掉  →  F 成立  →  无素参
      上尾停在某个平台上  →  残余存活    →  F 不成立

两个对照（缺了它们，任何结论都不可信）
--------------------------------------
  · **正对照**：把基底换成【同一臂的其余文档】。若 s 不显著变小，说明统计量根本不敏感，
    那么本次结果无论是什么都作废
  · **负对照**：把基底换成【随机打乱后的一批】，应给出 s ≈ 1

Run:  python PACSP-M/exploration/g1_F_test.py
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
CACHE = M / "onnx" / "embcache"


def unit(E):
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)


def pacsp_deltas(E):
    """The shipped increment definition, reimplemented so this script stays standalone."""
    return [float(np.linalg.norm(E[k + 1] - E[k])) for k in range(len(E) - 1)]


def load_pool():
    """Every arm that has a gpu_clsn_* cache. Returns (matrix, arm_label_per_row, arms)."""
    blocks, labels = [], []
    for p in sorted(CACHE.glob("gpu_clsn_*.npy")):
        arm = p.stem.replace("gpu_clsn_", "")
        E = unit(np.load(p))
        blocks.append(E)
        labels += [arm] * len(E)
    if not blocks:
        return None, None, []
    return np.vstack(blocks), np.asarray(labels), sorted(set(labels))


def nn_outside(X, labels, arm, exclude_own_arm=True):
    """For each row belonging to `arm`, distance to the nearest row outside that arm."""
    own = labels == arm
    sub = X[own]
    ext = X[~own] if exclude_own_arm else X
    if len(ext) == 0:
        return np.full(len(sub), np.nan)
    return np.linalg.norm(sub[:, None, :] - ext[None, :, :], axis=2).min(axis=1)


def nn_within(X, labels, arm):
    """Positive control: distance from each row to the nearest OTHER row of the same arm."""
    own = labels == arm
    sub = X[own]
    if len(sub) < 2:
        return np.full(len(sub), np.nan)
    D = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return D.min(axis=1)


def main() -> int:
    X, labels, arms = load_pool()
    if X is None:
        print("no gpu_clsn_* caches found", file=sys.stderr)
        return 2

    rng = np.random.default_rng(23)
    n = len(X)
    print("命题 F · 认真的检验\n")
    print(f"  池：{len(arms)} 个臂 / {n} 个文档")
    for a in arms:
        print(f"    {a:<26} {int((labels == a).sum()):>4} 篇")
    print()

    # ---------- the scale, from the data itself ---------------------------
    # The scale must match the ORDER STATISTIC being measured. r(d) is a MINIMUM over ~300
    # candidates, so comparing it to the median PAIRWISE distance is comparing a minimum to a
    # median — the negative control caught exactly this (it returned 0.55, not the ~1 I had
    # asserted). The right unit is the pool's own typical NEAREST-NEIGHBOUR distance.
    Dfull = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(Dfull, np.inf)
    D_nn = float(np.median(Dfull.min(axis=1)))
    i = rng.integers(0, n, size=20000)
    j = rng.integers(0, n, size=20000)
    keep = i != j
    D_pair = float(np.median(np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1)))
    print(f"  尺度 D_nn（池的典型最近邻距离，**归一化用这个**）= {D_nn:.4f}")
    print(f"  参考 D_pair（随机文档对的中位距离，**不用它归一化**）= {D_pair:.4f}")
    print(f"  〔两个数都是池自己给出的，不是我选的〕")
    print(f"  〔D_nn << D_pair 是序统计的必然：最小值 vs 中位数。用错的那个会让负对照偏离 1〕\n")

    # ---------- negative control: random substrate should give s slightly > 1 -----
    print("=" * 78)
    print("负对照：基底换成池中随机一半（候选变少 → 最小值变大 → 预期 s 略 > 1）")
    print("=" * 78)
    half = rng.permutation(n)[: n // 2]
    rest = np.setdiff1d(np.arange(n), half)
    r_rand = np.linalg.norm(X[half][:, None, :] - X[rest][None, :, :], axis=2).min(axis=1)
    s_rand = r_rand / D_nn
    print(f"  s 中位 {np.median(s_rand):.4f}   均值 {s_rand.mean():.4f}   "
          f"5% {np.percentile(s_rand,5):.4f}   95% {np.percentile(s_rand,95):.4f}")
    neg_ok = 0.9 <= float(np.median(s_rand)) <= 1.6
    print(f"  〔预期略 > 1；实测 {np.median(s_rand):.4f} → "
          f"{'✅ 校准通过' if neg_ok else '❌ 仍未校准，结论不可用'}〕\n")

    # ---------- positive control: same-arm substrate ----------------------
    print("=" * 78)
    print("正对照：基底换成【同一臂的其余文档】（若 s 不显著变小，本次全部作废）")
    print("=" * 78)
    print(f"  {'臂':<26}{'s 中位':>9}{'s 均值':>9}   预期：明显 < 1")
    pos = {}
    for a in arms:
        r = nn_within(X, labels, a)
        s = r / D_nn
        pos[a] = {"median": round(float(np.nanmedian(s)), 4),
                  "mean": round(float(np.nanmean(s)), 4)}
        print(f"  {a:<26}{np.nanmedian(s):>9.4f}{np.nanmean(s):>9.4f}")
    pos_all = [v["median"] for v in pos.values()]
    print(f"\n  正对照 s 中位的范围：{min(pos_all):.4f} – {max(pos_all):.4f}")
    sensitive = max(pos_all) < 0.9
    print(f"  统计量敏感？{'✅ 是（同臂基底明显更近）' if sensitive else '❌ 否——本次结论作废'}")
    print()

    # ---------- the actual test: external substrate, all arms -------------
    print("=" * 78)
    print("正式检验：基底 = 【全部其它臂】（真正外部）")
    print("=" * 78)
    print(f"  {'臂（个体）':<26}{'s 中位':>9}{'s 均值':>9}{'s 最大':>9}"
          f"{'s<0.5 占比':>12}")
    rows = {}
    for a in arms:
        r = nn_outside(X, labels, a)
        s = r / D_nn
        rows[a] = {"median": round(float(np.median(s)), 4),
                   "mean": round(float(s.mean()), 4),
                   "max": round(float(s.max()), 4),
                   "frac_lt_0.5": round(float((s < 0.5).mean()), 4),
                   "frac_lt_1.0": round(float((s < 1.0).mean()), 4)}
        print(f"  {a:<26}{np.median(s):>9.4f}{s.mean():>9.4f}{s.max():>9.4f}"
              f"{(s<0.5).mean():>12.4f}")

    all_s = np.concatenate([nn_outside(X, labels, a) / D_nn for a in arms])
    print(f"\n  全体 {len(all_s)} 篇的 s：")
    print(f"    中位 {np.median(all_s):.4f}   均值 {all_s.mean():.4f}   "
          f"最小 {all_s.min():.4f}   最大 {all_s.max():.4f}")
    print(f"    s < 0.5（有明显外部近邻）：{(all_s<0.5).mean()*100:.1f}%")
    print(f"    s < 1.0（比随机文档更近）：{(all_s<1.0).mean()*100:.1f}%")
    print(f"    s > 1.0（**比随机文档还孤立**）：{(all_s>1.0).mean()*100:.1f}%")

    # ---------- growth: does adding arms drive s down? --------------------
    print()
    print("=" * 78)
    print("F 的关键：逐步扩大基底，上尾是否向 0 收缩？")
    print("=" * 78)
    print(f"  {'基底臂数':<12}{'基底文档数':>12}{'s 中位':>10}{'s 上尾(95%)':>14}"
          f"{'s 最大':>10}")
    growth = []
    for k in range(1, len(arms)):
        for held in arms:
            sub_arms = [a for a in arms if a != held]
            # use the first k of the other arms as substrate
            selected = sub_arms[:k]
            mask = np.isin(labels, selected)
            if mask.sum() == 0:
                continue
            sub = X[mask]
            own = labels == held
            r = np.linalg.norm(X[own][:, None, :] - sub[None, :, :], axis=2).min(axis=1)
            s = r / D_nn
            growth.append({"k": k, "n_sub": int(mask.sum()), "held": held,
                           "median": float(np.median(s)),
                           "p95": float(np.percentile(s, 95)),
                           "max": float(s.max())})
    for k in range(1, len(arms)):
        g = [x for x in growth if x["k"] == k]
        if not g:
            continue
        print(f"  {k:<12}{int(np.mean([x['n_sub'] for x in g])):>12}"
              f"{np.mean([x['median'] for x in g]):>10.4f}"
              f"{np.mean([x['p95'] for x in g]):>14.4f}"
              f"{np.mean([x['max'] for x in g]):>10.4f}")

    print()
    print("  判定")
    k_med = {k: np.mean([x["median"] for x in growth if x["k"] == k])
             for k in range(1, len(arms))}
    ks = sorted(k_med)
    print(f"    k=1 时 s 中位 {k_med[ks[0]]:.4f} → k={ks[-1]} 时 {k_med[ks[-1]]:.4f}")
    drop = (k_med[ks[0]] - k_med[ks[-1]]) / k_med[ks[0]] if k_med[ks[0]] else 0.0
    print(f"    相对下降 {drop*100:.1f}%")
    if k_med[ks[-1]] < 0.2:
        print("    → 基底扩大把残余驱到很低：**F 在本次池内成立** → 无素参")
    elif drop < 0.15:
        print("    → 基底扩大几乎不改变 s：**存在不随基底扩大的残余** → F 不成立")
    else:
        print("    → 有下降但未驱到 0：**残余缩小但存活** → F 未成立，且下降是渐近的")

    print()
    print("=" * 78)
    print("决定性的一步：残余作为【尺度无关】的对象存在吗？")
    print("=" * 78)
    print("  「吸收」总需要一个尺度 ε：落在基底 ε-邻域内算被吸收。于是")
    print("      residue(ε) = Λ 中距基底 > ε 的那部分质量占比")
    print("  这个函数对 ε 单调不增，且 ε 足够大时归零。")
    print("  若如此，则 **⋂_{ε>0}(未被吸收的部分) = ∅**——")
    print("  尺度无关的「残余」**可证明为空**，不需要任何实验。")
    print()
    print(f"  {'臂（个体）':<26}" + "".join(f"{'ε='+f'{e:.1f}':>9}" for e in
                                            (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0)))
    mono_ok = True
    for a in arms:
        own = labels == a
        E = X[own]
        d = np.asarray(pacsp_deltas(E), dtype=float)
        pts = E[1:]
        ext = X[~own]
        dist = np.linalg.norm(pts[:, None, :] - ext[None, :, :], axis=2).min(axis=1)
        m = min(len(dist), len(d))
        tot = float(d[:m].sum())
        row = []
        prev = None
        for eps in (0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0):
            frac = float(d[:m][dist[:m] > eps].sum()) / tot if tot else 0.0
            row.append(frac)
            if prev is not None and frac > prev + 1e-12:
                mono_ok = False
            prev = frac
        print(f"  {a:<26}" + "".join(f"{v:>9.4f}" for v in row))
    print()
    print(f"  单调不增？{'✅ 全部成立' if mono_ok else '❌ 有违反'}")
    print("  ε = 2.0（单位球上的最大可能距离）时，残余必然为 0——")
    print("  这不是测量结果，是**单位球直径的上界**。")
    print()
    print("  ⇒ **尺度无关的「残余」不存在。**")
    print("    对任何固定的 ε，残余都是 ε 的函数；对 ε 取交集，得到空集。")
    print("    所以「路径中未被吸收的部分」这个说法**必然依赖一个尺度选择**，")
    print("    而尺度是实验者的自由选择——这正是前三次失败的共同根因，")
    print("    也是它**不可能**被第四次设计修好的原因。")
    print()
    print("  ⇒ 于是**命题 F 是空洞的**：")
    print("    F =「存在有限外部基底序列把残余驱到 0」")
    print("    若 ε 可以取大 → F 平凡成立，什么也没说")
    print("    若 ε 固定   → F 的真假完全由 ε 决定，不是关于世界的事实")

    out = M / "records_centroid" / "g1_F_test.json"
    out.write_text(json.dumps({
        "pool": {"arms": arms, "documents": n,
                 "D_nn": round(D_nn, 4), "D_pair": round(D_pair, 4)},
        "negative_control": {"median": round(float(np.median(s_rand)), 4)},
        "positive_control": pos, "sensitive": bool(sensitive),
        "external": rows,
        "pooled": {"median": round(float(np.median(all_s)), 4),
                   "mean": round(float(all_s.mean()), 4),
                   "min": round(float(all_s.min()), 4),
                   "max": round(float(all_s.max()), 4),
                   "frac_lt_0.5": round(float((all_s < 0.5).mean()), 4),
                   "frac_gt_1.0": round(float((all_s > 1.0).mean()), 4)},
        "growth": growth,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
