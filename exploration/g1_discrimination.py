"""G1 follow-up: does the more stable variant still DISCRIMINATE?

A metric that is order-invariant because it has stopped responding to the text is worthless.
``rn_selfnorm`` cut the mean |bias| from 0.1399 to 0.0967, so before calling that a win the
question has to be asked: is it stable because it is robust, or because it went blind?

Two checks, both against the shipped metric (``window``):

  1. Spearman rank correlation of the six arm values against the shipped metric's six.
     Near 1 means it still orders the corpora the same way.
  2. The paired HC3 contrast (AI medicine vs human medicine), which is the one arm pair in
     this set that is a genuine human/machine comparison rather than two different genres.

Run:  python PACSP-M/exploration/g1_discrimination.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE.parent
SRC = M / "records_centroid" / "g1_rn_density.json"

VARIANTS = ["global", "window", "narrow", "adjacent", "none",
            "rn_uniform", "rn_iso", "rn_centroid", "rn_selfnorm"]


def spearman(a: list[float], b: list[float]) -> float:
    """Rank correlation without scipy, ties broken by position."""
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da and db else 0.0


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}; run exploration/g1_rn_density.py first", file=sys.stderr)
        return 2
    d = json.loads(SRC.read_text(encoding="utf-8"))
    arms = [a for a in d if not a.startswith("_")]
    summary = d.get("_summary", {})

    print("G1 · 判别力检查：更稳的那个还看得见东西吗？\n")
    print(f"  臂 {len(arms)} 个：{', '.join(arms)}")
    print(f"  as-filed C_T 由各变体给出，参比是现行实现 window\n")

    ref = [d[a]["window"]["ct"] for a in arms]
    print(f"  {'variant':<14}{'mean|bias|':>12}{'Spearman vs window':>20}"
          f"{'AI/human (hc3)':>17}  判定")
    rows = []
    for v in VARIANTS:
        vals = [d[a][v]["ct"] for a in arms]
        rho = spearman(vals, ref)
        mab = summary.get("mean_abs_bias", {}).get(v)
        try:
            h = d["hc3_human_medicine"][v]["ct"]
            m_ = d["hc3_ai_medicine"][v]["ct"]
            ratio = m_ / h if h else float("nan")
        except KeyError:
            ratio = float("nan")
        # a variant is only worth adopting if it is BOTH more stable and still discriminating
        stable = (mab is not None) and (mab < summary["mean_abs_bias"]["none"])
        keeps_order = rho >= 0.9
        verdict = ("✅ 更稳且保序" if stable and keeps_order
                   else "⚠ 更稳但丢序" if stable
                   else "—")
        rows.append({"variant": v, "mean_abs_bias": mab, "spearman": round(rho, 4),
                     "hc3_ratio": round(ratio, 4) if ratio == ratio else None,
                     "stable_vs_none": bool(stable), "keeps_order": bool(keeps_order)})
        print(f"  {v:<14}{(mab if mab is not None else float('nan')):>12.4f}"
              f"{rho:>20.4f}{ratio:>17.4f}  {verdict}")

    print()
    print("  说明")
    print("    mean|bias| ：顺序敏感度，越小越稳（现行 window 0.1918，μ≡1 0.1399）")
    print("    Spearman   ：与现行实现的臂序一致性，接近 1 表示看到的排序没变")
    print("    AI/human   ：hc3 医学语料里 AI ÷ 人类，这是本组唯一的真·人机配对")
    print()
    winners = [r for r in rows if r["stable_vs_none"] and r["keeps_order"]]
    if winners:
        print(f"  同时满足「比 μ≡1 更稳」与「臂序不变」的："
              f"{', '.join(r['variant'] for r in winners)}")
        for r in winners:
            print(f"    {r['variant']}: |bias| {r['mean_abs_bias']:.4f}，"
                  f"Spearman {r['spearman']:.4f}，AI/human {r['hc3_ratio']}")
    else:
        print("  没有变体同时满足两个条件。")

    (M / "records_centroid" / "g1_discrimination.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {M / 'records_centroid' / 'g1_discrimination.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
