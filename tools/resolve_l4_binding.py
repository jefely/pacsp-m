"""Resolve what L4 actually anchors.

Reading pacsp_timestamp.py settled where the hash comes from: layer_4_timestamp takes
l1_result["data"]["content_hash"], not any L3 root, which is why matching L4's value
against the L3 roots failed.

That leaves a second question. The module writes the hash followed by a newline into a file
and runs `ots stamp` on that file, and OpenTimestamps timestamps the file's own SHA-256.
So the value submitted to the calendars is digest(content_hash + newline), one hop beyond
the content_hash recorded in the record. If so, the chain is

    corpus files -> L1.content_hash -> digest(that + newline) -> calendar -> (Bitcoin)

and the record stores the middle link while the calendars hold the last one, which makes
the binding derivable but not obvious.

This tests the hypothesis rather than asserting it.
"""

import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")


def main():
    recs = sorted((ID / "records").glob("*.pacsp"))
    print(f"  records: {len(recs)}")
    print(f"\n  {'record':<46} {'L4 == L1?':>10} {'digest(L1+nl)':>16} "
          f"{'L4[:16]':>18}")

    all_match_l1 = 0
    total = 0
    for p in recs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        l1 = rec.get("integrity", {}).get("L1", {})
        l4 = rec.get("integrity", {}).get("L4", {})
        l1d = (l1.get("data") or {})
        l1h = l1d.get("content_hash")
        l4h = ((l4.get("data") or {}).get("content_hash"))
        if not l1h or not l4h:
            print(f"  {p.stem[:44]:<46} {'n/a':>10}")
            continue
        total += 1
        same = l1h == l4h
        if same:
            all_match_l1 += 1
        hf = l4.get("data", {}).get("hash_file", "")
        digest = "sha256:" + hashlib.sha256(
            (l4h + "\n").encode("utf-8")).hexdigest()
        print(f"  {p.stem[:44]:<46} {str(same):>10} {digest[:16]:>16} {l4h[:16]:>18}")

    print(f"\n  L4.content_hash equals L1.content_hash in {all_match_l1}/{total} records")

    # The hash_file name is derived from the *L1* hash in the module; confirm the file exists
    print("\n  checking the stamped input file:")
    for p in recs[:3]:
        rec = json.loads(p.read_text(encoding="utf-8"))
        l4 = rec.get("integrity", {}).get("L4", {})
        hf = (l4.get("data") or {}).get("hash_file")
        if not hf:
            continue
        f = Path(hf)
        print(f"    {f.name}  exists={f.exists()}")
        if f.exists():
            content = f.read_text(encoding="utf-8")
            print(f"      content      : {content!r}")
            print(f"      digest of it : sha256:"
                  f"{hashlib.sha256(content.encode()).hexdigest()[:32]}")

    print("\n  interpretation:")
    if all_match_l1 == total and total:
        print("    L4 anchors the L1 content hash. The value in the record is not the")
        print("    value the calendars received: the calendars received the digest of")
        print("    that hash plus a newline, because ots stamps a file and the file held")
        print("    the hash text. The binding is therefore derivable, but only if the")
        print("    extra hop is known, and nothing in the record states it.")
        print("    Fix: record the submitted digest alongside, or stamp the raw bytes of")
        print("    the hash rather than its text.")
    else:
        print("    L4 does not simply mirror L1 for every record; see the table above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
