"""Identify the stamped object by matching cache files, not by parsing OTS internals.

The previous attempt guessed where the attested digest sits inside a detached proof and was
wrong in all nine records, so parsing position is not a reliable route. The cache directory
holds the actual stamped inputs and their proofs, named by the first 16 hex characters of
the digest that was stamped. That naming makes the question answerable directly: compute
each candidate digest for a record and see which one names an existing file.

Candidates considered:
  raw hex            the 64 hex characters, no newline
  raw hex + LF       what the module's open().write() would produce from the hash alone
  sha256:hex + LF    what it produces when content_hash includes the algorithm prefix
  sha256(hex text)   the digest of that text, which is what ots actually receives
  sha256(hex + LF)
  sha256(sha256:hex)
  sha256(sha256:hex + LF)
"""

import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
OTS = ID / "cache" / "ots"


def candidates(content_hash: str) -> dict:
    hexpart = content_hash.split(":", 1)[1]
    out = {}
    texts = {
        "hex": hexpart,
        "hex+LF": hexpart + "\n",
        "sha256:hex": content_hash,
        "sha256:hex+LF": content_hash + "\n",
    }
    for name, t in texts.items():
        out[f"sha256({name})"] = hashlib.sha256(t.encode("utf-8")).hexdigest()
    out["hex-bytes"] = hexpart
    out["sha256(raw-bytes)"] = hashlib.sha256(bytes.fromhex(hexpart)).hexdigest()
    return out


def main():
    existing = {p.stem: p for p in OTS.glob("*.txt")}
    print(f"  cache .txt files: {len(existing)}")
    print(f"  sample names    : {sorted(existing)[:4]}")

    recs = sorted((ID / "records").glob("*.pacsp"))
    print(f"\n  resolving each record's stamped object")
    resolved = {}
    for p in recs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = rec.get("integrity", {}).get("L4", {}).get("data") or {}
        l4h = d.get("content_hash")
        if not l4h:
            continue
        cands = candidates(l4h)
        hits = {k: v for k, v in cands.items()
                if v[:16] in existing or v in existing}
        resolved[p.stem] = hits
        label = ", ".join(sorted(hits)) if hits else "NO MATCH"
        print(f"    {p.stem[:44]:<46} {label}")

    print("\n  which candidate explains the most records:")
    tally = {}
    for hits in resolved.values():
        for k in hits:
            tally[k] = tally.get(k, 0) + 1
    for k, v in sorted(tally.items(), key=lambda x: -x[1]):
        print(f"    {k:<24} {v}/{len(resolved)} records")

    print("\n  reading a stamped input file to confirm:")
    for name in sorted(existing)[:2]:
        t = existing[name].read_text(encoding="utf-8")
        print(f"    {name}.txt  {len(t)} chars  {t!r}")
        print(f"      sha256 of its content: "
              f"{hashlib.sha256(t.encode()).hexdigest()[:16]}")

    print("\n  interpretation:")
    if tally and max(tally.values()) == len(resolved):
        best = max(tally, key=tally.get)
        print(f"    Every record resolves to the same construction: {best}.")
        print("    So the chain is derivable from the record plus the cache, but the")
        print("    record does not state which construction was used, and the cache is")
        print("    not part of the record. A verifier with only the .pacsp file cannot")
        print("    reach the calendars' digest without guessing.")
    else:
        print("    No single construction explains all records; see the tally above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
