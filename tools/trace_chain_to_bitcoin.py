"""Trace the full chain from the corpus to the Bitcoin block, now that the proof is upgraded.

`ots info` reports the file hash it attests as 26e56810..., while the record stores
L4.content_hash as 71b1c5f6.... The earlier audit concluded the binding could not be recovered
from the record alone, and the upgrade did not change that; what it changes is that a Bitcoin
anchor now exists at all.

This walks the chain explicitly: the corpus, the L1 content hash, the intermediate values, and
finally the digest the upgraded proof attests. If the record's hash can be shown to reach the
proof's hash through a documented sequence, the binding is derivable in practice even though
the record does not spell it out.
"""

import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
M = Path(r"D:\myproject\PACSP-M")
WORK = M / "records_centroid" / "ots_upgrade"
sys.path.insert(0, str(ID / "scripts"))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    rec_path = ID / "records" / "human_poem_epoch1_base_CT11.54Se_20261004.pacsp"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    l4 = rec["integrity"]["L4"]["data"]
    content_hash = l4["content_hash"]
    hexpart = content_hash.split(":", 1)[1]

    print(f"  record                 : {rec_path.name}")
    print(f"  L1/L4 content_hash     : {content_hash}")

    # what the file naming tells us
    print(f"\n  ots_file recorded      : {Path(l4['ots_file']).name}")
    print(f"  hash_file recorded     : {l4.get('hash_file')}")

    # the tokenizer-style file that was stamped: cache/ots/<prefix>.txt
    cand = ID / "cache" / "ots" / f"{hexpart[:16]}.txt"
    print(f"\n  candidate stamped file : {cand.name}  exists={cand.exists()}")
    if cand.exists():
        txt = cand.read_text(encoding="utf-8")
        print(f"    content              : {txt!r}")
        print(f"    sha256(utf-8 bytes)  : {sha256_bytes(txt.encode('utf-8'))}")
        print(f"    sha256(raw bytes)    : {sha256_bytes(cand.read_bytes())}")

    # what the upgraded proof attests
    up = WORK / "71b1c5f6553d4371.txt.ots"
    print(f"\n  upgraded proof         : {up.name}  {up.stat().st_size} B")
    r = subprocess.run(["ots", "info", str(up)], capture_output=True, text=True,
                       timeout=300)
    first = next((ln for ln in r.stdout.splitlines() if "File sha256 hash" in ln), "")
    attested = first.split(":")[-1].strip()
    print(f"    attests              : {attested}")

    print("\n  === does the record's hash reach the attested hash? ===")
    tests = [
        ("sha256(hexpart)", sha256_bytes(hexpart.encode())),
        ("sha256(hexpart+LF)", sha256_bytes((hexpart + "\n").encode())),
        ("sha256('sha256:'+hexpart)", sha256_bytes(content_hash.encode())),
        ("sha256('sha256:'+hexpart+LF)",
         sha256_bytes((content_hash + "\n").encode())),
    ]
    for label, v in tests:
        hit = v == attested
        print(f"    {label:<34} {v[:16]}  {'<== MATCH' if hit else ''}")

    # the file whose sha256 is 26e56810... is whatever was stamped
    print(f"\n  searching the cache for a file whose sha256 is {attested[:16]}...")
    found = []
    for f in (ID / "cache" / "ots").glob("*.txt"):
        if sha256_bytes(f.read_bytes()) == attested:
            found.append(("raw bytes", f))
        if sha256_bytes(f.read_text(encoding="utf-8").encode()) == attested:
            found.append(("utf-8 text", f))
    for kind, f in found:
        print(f"    {f.name}  ({kind})")

    print("\n  === the three-generation picture ===")
    print("    generation 1: cache/ots/<hash16>.txt.ots, 840 B, pending")
    print("                  file stamped = <hash16>.txt as it then existed")
    print("    generation 2: the same, upgraded in place to 2913 B")
    print("                  now carries BitcoinBlockHeaderAttestation(969867) and (969869)")
    print("    the record points at generation 1 by path and stores the pre-upgrade")
    print("    timestamp_proof, so a verifier holding only the record sees pending")
    print("    attestations and cannot reach the chain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
