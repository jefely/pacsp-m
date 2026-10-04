"""Count block and inline formula occurrences in the paper source."""

import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
SRC = M / "PACSP-M-1.1.0.md"


def main():
    t = SRC.read_text(encoding="utf-8")
    lines = t.splitlines()

    blocks = []
    buf, inm = [], False
    for L in lines:
        s = L.strip()
        if s == "$$" or s == "\\[" or s == "\\]":
            if inm:
                if buf:
                    blocks.append(" ".join(buf))
                buf, inm = [], False
            else:
                inm = True
            continue
        if inm and s:
            buf.append(s)
    if buf:
        blocks.append(" ".join(buf))

    print(f"  block formulas: {len(blocks)}")
    for b in blocks:
        print(f"    {b[:120]}")

    inline = re.findall(r"(?<!\$)\$([^$\n]{2,80})\$(?!\$)", t)
    print(f"\n  inline $...$ spans: {len(inline)}")
    for x in sorted(set(inline))[:25]:
        print(f"    {x}")

    for tok in ("\\frac", "\\binom", "\\sum", "\\int", "\\sqrt", "\\text",
                "\\|", "\\{", "\\}", "\\mu", "\\sigma", "\\Phi", "\\Omega",
                "\\Lambda", "\\delta", "\\le", "\\ge", "\\times", "\\cdot"):
        n = t.count(tok)
        if n:
            print(f"  {tok:<10} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
