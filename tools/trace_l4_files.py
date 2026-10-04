"""Establish what each cached .txt holds, and whether it corresponds to the ots_file field.

The record's ots_file points at cache/ots/<first16>.txt.ots, but that name is not a prefix
of the record's content_hash, while the .txt files that do carry the content hash prefix are
a different set. So there appear to be two generations of files in the cache, and only one
of them matches what the records point at.

This reads both sets and reports what each actually contains, so the provenance is settled
rather than inferred.
"""

import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
OTS = ID / "cache" / "ots"


def main():
    txts = sorted(OTS.glob("*.txt"))
    print(f"  .txt files in cache: {len(txts)}")
    print(f"\n  {'name':<20} {'len':>5}  content")
    contents = {}
    for t in txts:
        raw = t.read_text(encoding="utf-8")
        contents[t.stem] = raw
        print(f"  {t.stem:<20} {len(raw):>5}  {raw!r}")

    print("\n  for each record: does ots_file's stem appear among the cache .txt names?")
    recs = sorted((ID / "records").glob("*.pacsp"))
    match_by_stem = 0
    match_by_contenthead = 0
    for p in recs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = rec["integrity"]["L4"].get("data") or {}
        of = d.get("ots_file")
        ch = d.get("content_hash")
        if not of:
            continue
        stem = Path(of).name.replace(".txt.ots", "")
        hexpart = ch.split(":", 1)[1] if ch else ""
        in_cache = stem in contents
        head_match = contents.get(stem, "").strip() == ch
        match_by_stem += in_cache
        match_by_contenthead += head_match
        print(f"    {p.stem[:42]:<44} stem={stem[:16]} in_cache={in_cache} "
              f"content==hash={head_match}")
        if in_cache and not head_match:
            print(f"        cache content: {contents[stem]!r}")
            print(f"        record hash  : {ch!r}")

    print(f"\n  ots_file stems present in cache        : {match_by_stem}/{len(recs)}")
    print(f"  ... and their content equals the hash  : {match_by_contenthead}/{len(recs)}")

    # do the content-hash-prefixed files correspond to the records at all?
    print("\n  are there .txt files named after the content hash of each record?")
    for p in recs[:4]:
        rec = json.loads(p.read_text(encoding="utf-8"))
        ch = (rec["integrity"]["L4"].get("data") or {}).get("content_hash", "")
        pref = ch.split(":", 1)[1][:16] if ch else ""
        present = pref in contents
        print(f"    {p.stem[:42]:<44} prefix {pref} present={present}")

    print("\n  reading the ots_file for a record and reporting its .txt sibling:")
    for p in recs[:2]:
        rec = json.loads(p.read_text(encoding="utf-8"))
        of = (rec["integrity"]["L4"].get("data") or {}).get("ots_file")
        if not of:
            continue
        txt = Path(of).with_suffix("")  # strip .ots
        print(f"    {txt.name}: exists={txt.exists()}"
              + (f"  {txt.read_text(encoding='utf-8')!r}" if txt.exists() else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
