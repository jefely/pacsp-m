"""G1 · the framework's own Lebesgue decomposition, applied to the path measure.

The framework writes its measure as

    dLambda = rho dnu + dM          (rho = the RN density, M = the part singular to nu)

and calls nu the "public cognitive substrate" and M the un-absorbable residual (素参). G1's
claim is that mu should be rho, computed per cell. This script shows the decomposition
explicitly, using the project's own math tool rather than a restatement of it.

Two things are being demonstrated, and they are different in kind:

  * the RN density rho, which is what G1 proposes mu should BE
  * the absorbed / singular split, which says how much of the path a given substrate can
    account for — the framework's own language for 素参

Run:  python PACSP-M/exploration/g1_lebesgue_link.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
M = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(M.parent / "pacsp-math-py"))

from g1_rn_density import ARMS, CACHE, delta_of, reference  # noqa: E402

try:
    from pacsp_math.lebesgue import lebesgue_decompose
except Exception as exc:                                       # noqa: BLE001
    print(f"pacsp_math.lebesgue unavailable: {exc}", file=sys.stderr)
    raise SystemExit(2)


def main() -> int:
    print("G1 · 框架自己的 dΛ = ρ dν + dM，跑在路径测度上\n")
    print(f"  {'arm':<22}{'Σδ (=Λ总量)':>13}{'ν 用哪种':>26}{'被吸收':>10}{'奇异':>10}")
    for arm in ARMS:
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if not p.exists():
            continue
        E = np.load(p)
        d = delta_of(E)

        for kind, label in (("rn_uniform", "计数测度 ν=1"),
                            ("rn_centroid", "到质心的距离（基底）")):
            nu, _ = reference(E, kind)
            m = min(len(d), len(nu))
            r = lebesgue_decompose(mu=[float(x) for x in d[:m]],
                                   nu=[float(x) for x in nu[:m]])
            # the Python module uses snake_case; the DSH tool exposes camelCase
            absorbed = r.get("absorbed_fraction", r.get("absorbedFraction"))
            sing = r.get("singular_fraction", r.get("singularFraction"))
            print(f"  {arm:<22}{float(np.sum(d)):>13.4f}{label:>26}"
                  f"{(absorbed if absorbed is not None else float('nan')):>12.4f}"
                  f"{(sing if sing is not None else float('nan')):>12.4f}")

    print()
    print("  ⚠ 这个脚本**不能**用来回答「素参存不存在」。")
    print()
    print("  它上面那张表的奇异部分恒为 0，是**构造上必然的**，不是测量结果：")
    print("  有限分割上，勒贝格奇异部分 = ν=0 的格子上的质量。")
    print("  而这里用的 ν（计数测度、到质心距离）**处处为正**，")
    print("  所以 dM = 0 不可能有别的结果。")
    print()
    print("  它唯一说明的是：**ν 取什么，决定了奇异部分有多大**——")
    print("  这恰恰意味着「素参有多大」是 ν 的函数，而不是一个可测的量。")
    print()
    print("  要看素参在**真正外来的基底**下是否非零，见：")
    print("    python exploration/g1_su_parameter_exists.py")
    print()
    print("  〔留档〕这个脚本的早期版本把上表读成了「素参恒为 0」的结论，")
    print("   那个结论是错的，已撤回。错误在于用了一个不可能给出反例的检验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
