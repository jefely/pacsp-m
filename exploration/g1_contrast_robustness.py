"""G1 decisive test: is the shipped metric's stronger human/AI contrast real, or is it
the order sensitivity showing through?

The tradeoff G1 exposed:

    window (shipped)   order sensitivity 0.1918   AI/human contrast 1.296
    none   (mu = 1)    order sensitivity 0.1399   AI/human contrast 1.085
    rn_selfnorm        order sensitivity 0.0967   AI/human contrast 1.085

The more stable variants also show a WEAKER contrast. That is not automatically a loss: if
the shipped metric's extra contrast comes from where the items happened to sit in the file,
it is not signal. The test is whether the contrast survives reordering.

Method. The hc3 pair (human medicine, AI medicine) is the one genuine human/machine
comparison in this arm set. For each variant, recompute the ratio after independently
permuting BOTH arms, and look at the distribution. A contrast that is real barely moves; a
contrast built on order collapses toward 1.

Run:  python PACSP-M/exploration/g1_contrast_robustness.py
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
from g1_rn_density import ARMS, CACHE, RN, ct_of  # noqa: E402

NPERM = 200
HUMAN, AI = "hc3_human_medicine", "hc3_ai_medicine"
VARIANTS = ["window", "global", "none", "rn_uniform", "rn_selfnorm"]


def main() -> int:
    rng = np.random.default_rng(11)
    E = {}
    for arm in (HUMAN, AI):
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if not p.exists():
            print(f"missing cache for {arm}", file=sys.stderr)
            return 2
        E[arm] = np.load(p)

    print("G1 · 决定性检验：更强的人机对比，是信号还是顺序噪声？\n")
    print(f"  配对：{AI}  vs  {HUMAN}")
    print(f"  每次独立打乱两臂，各 {NPERM} 次\n")
    print(f"  {'variant':<14}{'as-filed':>10}{'shuf mean':>11}{'shuf sd':>9}"
          f"{'sd/mean':>9}{'>1 的比例':>11}  判定")

    out = {}
    for v in VARIANTS:
        a_h = ct_of(E[HUMAN], v)
        a_a = ct_of(E[AI], v)
        asis = a_a / a_h

        ratios = []
        for _ in range(NPERM):
            h = ct_of(E[HUMAN][rng.permutation(len(E[HUMAN]))], v)
            a = ct_of(E[AI][rng.permutation(len(E[AI]))], v)
            ratios.append(a / h)
        ratios = np.asarray(ratios)
        mean, sd = float(ratios.mean()), float(ratios.std())
        frac = float((ratios > 1).mean())
        # the contrast is robust if it stays clear of 1 across reorderings
        if frac == 1.0:
            verdict = "✅ 稳健（每次重排都 > 1）"
        elif frac >= 0.95:
            verdict = f"✅ 基本稳健（{frac*100:.0f}% > 1）"
        elif frac >= 0.75:
            verdict = f"⚠ 弱（仅 {frac*100:.0f}% > 1）"
        else:
            verdict = f"❌ 不成立（仅 {frac*100:.0f}% > 1）"
        out[v] = {"as_filed": round(asis, 4), "shuf_mean": round(mean, 4),
                  "shuf_sd": round(sd, 4), "sd_over_mean": round(sd / mean, 4),
                  "frac_above_1": round(frac, 4), "robust": frac >= 0.95}
        print(f"  {v:<14}{asis:>10.4f}{mean:>11.4f}{sd:>9.4f}"
              f"{sd/mean:>9.4f}{frac:>11.3f}  {verdict}")

    print()
    print("  读法")
    print("    as-filed   ：按文件里的实际顺序算出的比值")
    print("    shuf mean  ：独立打乱两臂后的比值均值——**如果对比是真的，它应接近 as-filed**")
    print("    sd/mean    ：比值本身的不确定度；越小说明这个差越不依赖顺序")
    print("    >1 的比例  ：重排后「AI 高于人类」仍然成立的比例")
    print()
    rob = [v for v in VARIANTS if out[v]["robust"]]
    print(f"  对比在重排下仍然成立的变体：{rob if rob else '无'}")
    print()
    print("  结论用法")
    print("    若 window 不稳健 → 它那多出来的对比是顺序伪影，不该当作判别力")
    print("    若 window 也稳健   → 则「更稳 = 更弱」是真实的权衡，需要你来做取舍")

    (M / "records_centroid" / "g1_contrast_robustness.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {M / 'records_centroid' / 'g1_contrast_robustness.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
