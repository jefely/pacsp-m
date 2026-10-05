"""Smoke-check the exact code path the web interface calls, for the corrected gate.

The browser runs pacsp_serve.do_compare, so that is what has to be checked before telling
anyone to reload the page. Running the CLI proves the library works; it does not prove the
interface will show the right thing.
"""

import sys
from pathlib import Path

M = Path(r"D:\myproject\PACSP-M")
sys.path.insert(0, str(M))

import pacsp_serve as S  # noqa: E402

PAIRS = [
    ("data/poem", "data/machine_poem", "poem"),
    ("data/lyrics", "data/machine_lyrics", "lyrics"),
    ("data/techdoc", "data/machine_techdoc2", "techdoc"),
    ("data/hc3_human_medicine", "data/hc3_ai_medicine", "medicine"),
    ("data/hc3_human_openqa", "data/hc3_ai_openqa", "openqa"),
]


def main():
    print(f"  {'pair':<10} {'overlap':>8} {'accuracy':>9} {'wrong':>7}  verdict")
    bad = []
    for a, b, dom in PAIRS:
        pa, pb = M / a, M / b
        if not pa.is_dir() or not pb.is_dir():
            print(f"  {dom:<10} corpus missing")
            continue
        d = S.do_compare(str(pa), str(pb), "bge-large-zh", "onnx", 2000)
        ov = d["overlap"]["value"]
        acc = d["assignability"]["accuracy"]
        wrong = d["assignability"]["negative_margins"]
        n = d["assignability"]["n"]
        v = d["verdict"].split(":")[0]
        print(f"  {dom:<10} {ov:>8.4f} {acc:>9.4f} {wrong:>3}/{n:<3}  {v}")
        # the corrected behaviour: a pair clearly above chance must not be called
        # unassignable, and the verdict must be separable
        if acc >= 0.6 and v != "separable":
            bad.append(f"{dom}: accuracy {acc:.3f} but verdict {v}")
        if "assignability" not in d:
            bad.append(f"{dom}: assignability missing from the payload")

    print()
    if bad:
        print("  PROBLEMS:")
        for x in bad:
            print(f"    - {x}")
        return 1
    print("  the web path reports the corrected verdict for every pair")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
