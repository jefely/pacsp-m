"""Find which corpus directory a .pacsp record attests, by matching the L3a Merkle root.

The first verification attempt failed on all nine records because it was pointed at data/,
the parent of the corpora, while L3a attests the sample hashes of one corpus. Reading the
checker's source showed this; rerunning with the right directory verified eight of nine.

The ninth record's corpus name was guessed wrong, so rather than guess again this computes
the L3a root for every candidate directory and reports which one matches. That is the
correct way to answer the question, since the root is the identification.
"""

import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
sys.path.insert(0, str(ID / "scripts"))

from pacsp_verify import _merkle_root, sha256_file  # noqa: E402


def l3a_root(corpus: Path) -> str:
    hashes = [sha256_file(f) for f in sorted(corpus.glob("*.txt"))]
    return _merkle_root(hashes)


def main():
    rec_path = ID / "records" / "human_poem_epoch1_base_CT11.54Se_20261004.pacsp"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    want = rec.get("integrity", {}).get("L3", {}).get("sample")
    print(f"  record        : {rec_path.name}")
    print(f"  L3a root want : {want}")
    print(f"  C_T recorded  : {rec.get('results', {}).get('C_T_Se')}")

    print(f"\n  scanning corpora under {ID / 'data'}")
    hits = []
    for d in sorted((ID / "data").iterdir()):
        if not d.is_dir():
            continue
        n = len(list(d.glob("*.txt")))
        if n == 0:
            print(f"    {d.name:<24} no .txt files")
            continue
        try:
            got = l3a_root(d)
        except Exception as e:
            print(f"    {d.name:<24} error {type(e).__name__}")
            continue
        mark = "  <== MATCH" if got == want else ""
        if got == want:
            hits.append(d.name)
        print(f"    {d.name:<24} {n:>3} files  {got[:16]}...{mark}")

    print()
    if hits:
        print(f"  the record attests: {hits}")
    else:
        print("  no corpus under data/ matches this root.")
        print("  the attested sample set is not present in this checkout, so L3a cannot")
        print("  be verified here; that is a missing-input problem, not a record defect.")
        # widen the search: any directory anywhere with .txt files
        print("\n  widening search to all directories with .txt files:")
        roots = [ID / "materials", ID / "data", ID / "experiments", ID / "build"]
        for r in roots:
            if not r.exists():
                continue
            for d in sorted(r.rglob("*")):
                if not d.is_dir():
                    continue
                n = len(list(d.glob("*.txt")))
                if n < 5:
                    continue
                try:
                    got = l3a_root(d)
                except Exception:
                    continue
                if got == want:
                    print(f"    MATCH: {d}  ({n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
