"""Confirm the frozen result snapshots match a fresh run of the scripts.

results/ holds the numbers the documents quote. records_centroid/ is where the scripts
write when run. If a rerun reproduces the frozen snapshot exactly, the analysis is
reproducible from this folder alone; if it differs, either the environment moved or the
frozen values were not produced by the code as it now stands, and either way that should
be known rather than assumed.
"""

import hashlib
import json
from pathlib import Path

M = Path(__file__).resolve().parent
FROZEN = M / "results"
FRESH = M / "records_centroid"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    frozen = sorted(FROZEN.glob("*.json"))
    fresh = sorted(FRESH.glob("*.json"))
    print(f"  results/          {len(frozen)} 个")
    print(f"  records_centroid/ {len(fresh)} 个")
    if not fresh:
        print("\n  尚无新运行输出；先跑一个脚本再比对")
        return 0

    names = {p.name for p in frozen} & {p.name for p in fresh}
    print(f"\n  同名可比对: {len(names)} 个")
    print(f"  {'文件':<36} {'定稿':>9} {'本次':>9}  判定")
    same = diff = 0
    for n in sorted(names):
        a, b = FROZEN / n, FRESH / n
        sa, sb = sha(a), sha(b)
        try:
            equal = json.loads(a.read_text(encoding="utf-8")) == \
                    json.loads(b.read_text(encoding="utf-8"))
        except Exception:
            equal = sa == sb
        if equal:
            same += 1
        else:
            diff += 1
        print(f"  {n:<36} {sa:>9} {sb:>9}  "
              f"{'IDENTICAL' if equal else 'DIFFERS'}")

    print(f"\n  {same} 个一致, {diff} 个不同")
    if diff == 0:
        print("  RESULT: 可复现")
    else:
        print("  RESULT: 有差异，需检查")
    return 0 if diff == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
