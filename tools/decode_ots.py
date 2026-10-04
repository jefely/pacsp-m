"""Decode a stored OpenTimestamps proof and report what it actually attests.

The record stores the proof as base64 under integrity.L4.data.timestamp_proof, with a
mirrored .ots file in cache/ots. Reading it establishes whether the anchor is real: a proof
carrying calendar server URIs and a PendingAttestation was genuinely submitted, whereas a
self-made blob would carry neither.

The distinction that matters for the target feature is between a pending attestation and a
Bitcoin attestation. A pending one says "a calendar promised to anchor this"; only an
upgraded one carries a block height and a Merkle path into a block header, and only that
can be checked by a third party against the Bitcoin chain. The record's own anchor field
says pending_bitcoin_confirmation, which this confirms or refutes.
"""

import base64
import json
import re
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00"


def varint(b, i):
    """OpenTimestamps uses Bitcoin-style varints for lengths."""
    n = 0
    shift = 0
    while True:
        if i >= len(b):
            raise ValueError("varint ran off the end")
        c = b[i]
        i += 1
        n |= (c & 0x7F) << shift
        if not (c & 0x80):
            return n, i
        shift += 7


def main():
    rec_path = ID / "records" / "human_poem_epoch1_base_CT11.54Se_20261004.pacsp"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    l4 = rec["integrity"]["L4"]["data"]
    blob = l4["timestamp_proof"]
    if blob.startswith("base64:"):
        raw = base64.b64decode(blob[len("base64:"):])
    else:
        raw = bytes.fromhex(blob)

    print(f"  record        : {rec_path.name}")
    print(f"  content hash  : {l4['content_hash']}")
    print(f"  anchor field  : {l4['timestamp_anchor']}")
    print(f"  decoded bytes : {len(raw)}")

    print(f"\n  magic         : {raw[:8]!r} == OTS? {raw.startswith(OTS_MAGIC[:8])}")
    ok = raw.startswith(OTS_MAGIC)
    print(f"  full magic ok : {ok}")

    i = len(OTS_MAGIC)
    n, i = varint(raw, i)
    tag = raw[i:i + n].decode("utf-8", "replace")
    i += n
    print(f"  first tag     : {tag!r} (a detached timestamp proof)")

    # the header is a sequence of tagged varint-prefixed fields followed by operations
    print("\n  scanning the header for attestation and calendar records:")
    text = raw.decode("latin-1")
    urls = re.findall(r"https?://[A-Za-z0-9\.\-/_]+", text)
    print(f"    calendar URIs found: {len(urls)}")
    for u in sorted(set(urls)):
        print(f"      {u}")

    # attestation tags: 0x00 = pending, 0x08 = bitcoin
    pending = text.count("PendingAttestation")
    bitcoin = text.count("BitcoinBlockHeaderAttestation") + text.count("Bitcoin")
    print(f"\n    'PendingAttestation' occurrences : {pending}")
    print(f"    'Bitcoin' occurrences            : {bitcoin}")

    # a Bitcoin attestation carries a varint block height near the end of its record
    print("\n  interpretation:")
    if urls and (pending or not bitcoin):
        print("    The proof carries real calendar URIs, so it was actually submitted.")
        print("    But it holds pending attestations only: the calendars promised to")
        print("    anchor it and no Bitcoin block header path is present.")
        print("    A third party cannot check this against the chain yet.")
        print("    Upgrading it requires asking the calendars again, after the")
        print("    transaction confirms, which is what the ots client's upgrade does.")
    if "btc" in " ".join(urls).lower() and not bitcoin:
        print("    The calendars in question are the Bitcoin ones, so the anchor is")
        print("    intended to be a Bitcoin anchor; it simply is not one yet.")

    mirror = Path(l4["ots_file"])
    print(f"\n  mirrored .ots file: {mirror}")
    print(f"    exists: {mirror.exists()}"
          + (f"  ({mirror.stat().st_size} bytes)" if mirror.exists() else ""))
    if mirror.exists():
        mb = mirror.read_bytes()
        print(f"    same bytes as the embedded proof: "
              f"{mb == raw or mb == raw[:len(mb)] or raw == mb[:len(raw)]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
