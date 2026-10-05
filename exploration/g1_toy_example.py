"""G1 · the smallest possible worked example, so the mechanism is visible rather than asserted.

Four points on a line. Two functionals:

    total path length   sum(delta)          -- what mu = 1 computes
    mass-weighted step  sum(delta^2)/sum(delta)   -- what the faithful RN density computes

Visiting the same four points in a different order changes BOTH numbers. The question is by
how much, relatively. This is the whole of the order-sensitivity issue, small enough to check
by hand.

Run:  python PACSP-M/exploration/g1_toy_example.py
"""
from __future__ import annotations

import itertools
from pathlib import Path

PTS = [0.0, 1.0, 10.0, 11.0]
LABEL = {0.0: "A=0", 1.0: "B=1", 10.0: "C=10", 11.0: "D=11"}


def main() -> int:
    rows = []
    for perm in itertools.permutations(PTS):
        d = [abs(perm[i] - perm[i - 1]) for i in range(1, len(perm))]
        total = sum(d)
        sq = sum(x * x for x in d)
        rows.append({"order": "-".join("ABCD"[PTS.index(p)] for p in perm),
                     "deltas": [int(x) for x in d],
                     "sum": total, "sumsq": sq, "ratio": sq / total})

    print("G1 · 最小可核对的例子：4 个点，24 种顺序\n")
    print(f"  点：A=0  B=1  C=10  D=11   （故意让间距不均匀，好让顺序产生影响）\n")
    print(f"  {'顺序':<12}{'各段 δ':>16}{'Σδ':>8}{'Σδ²':>8}{'Σδ²/Σδ':>10}")
    for r in rows:
        print(f"  {r['order']:<12}{str(r['deltas']):>16}{r['sum']:>8.0f}"
              f"{r['sumsq']:>8.0f}{r['ratio']:>10.4f}")

    sums = [r["sum"] for r in rows]
    ratios = [r["ratio"] for r in rows]

    def cv(xs):
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return (var ** 0.5) / m

    print()
    print(f"  {'':<12}{'最小':>8}{'最大':>8}{'均值':>8}{'变异系数 CV':>14}")
    print(f"  {'Σδ':<12}{min(sums):>8.0f}{max(sums):>8.0f}"
          f"{sum(sums)/len(sums):>8.2f}{cv(sums):>14.4f}")
    print(f"  {'Σδ²/Σδ':<12}{min(ratios):>8.4f}{max(ratios):>8.4f}"
          f"{sum(ratios)/len(ratios):>8.4f}{cv(ratios):>14.4f}")

    print()
    print("  这就是机制本身：")
    print(f"    Σδ      在不同顺序下从 {min(sums):.0f} 变到 {max(sums):.0f}——**差 {max(sums)/min(sums):.1f} 倍**")
    print(f"    Σδ²/Σδ  只从 {min(ratios):.4f} 变到 {max(ratios):.4f}——**差 {max(ratios)/min(ratios):.2f} 倍**")
    print()
    print(f"    相对波动：{cv(sums):.4f}  vs  {cv(ratios):.4f}"
          f"   （比值稳了约 {cv(sums)/cv(ratios):.0f} 倍）")
    print()
    print("  为什么：Σδ² 与 Σδ 都随顺序一起变大变小，相除把共同的成分约掉了。")
    print("  这不是「把顺序敏感性调小」，而是**换了一个对顺序不敏感的量**——")
    print("  代价见 g1_contrast_robustness.py：人机对比也从 1.296 降到 1.085。")

    out = Path(__file__).resolve().parent.parent / "records_centroid" / "g1_toy_example.json"
    import json
    out.write_text(json.dumps({"rows": rows,
                               "cv_sum": round(cv(sums), 4),
                               "cv_ratio": round(cv(ratios), 4)},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
