"""Find unmatched inline markers in the markdown source.

The DOCX showed a literal ** in the rendered text, which means either the source has an
odd number of markers or the parser mis-split a span. This distinguishes the two: it
reports lines with an odd marker count, and lines where markers survive parsing.
"""

import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
SRC = M / "PACSP-M-1.1.0.md"
INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*|`[^`]+`)")


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    odd = []
    survives = []
    for i, L in enumerate(lines, 1):
        s = L.strip()
        if not s or s.startswith("```"):
            continue
        n = s.count("**")
        if n % 2 == 1:
            odd.append((i, s))
        parts = INLINE.split(s)
        # a marker survives if any piece that is not itself a matched span still holds it
        for p in parts:
            if p.startswith("**") and p.endswith("**") and len(p) > 4:
                continue
            if "**" in p:
                survives.append((i, s, p[:70]))
                break

    print(f"  lines with an odd count of '**': {len(odd)}")
    for i, s in odd[:20]:
        print(f"    L{i}: {s[:110]}")
    print(f"\n  lines where '**' survives parsing: {len(survives)}")
    for i, s, p in survives[:20]:
        print(f"    L{i}: frag={p!r}")
        print(f"         full={s[:110]}")

    print()
    for i, L in enumerate(lines, 1):
        if "机器更同质" in L or "经单篇层面" in L:
            print(f"  L{i}: {L!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
