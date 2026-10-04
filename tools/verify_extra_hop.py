"""Verify the extra-hop hypothesis about L4.

`ots stamp` timestamps a file, so what the calendars received is the SHA-256 of the file's
contents. The module wrote the content hash plus a newline into that file, so the calendars
hold digest("sha256:<hex>\\n") while the record stores "sha256:<hex>". A detached OTS proof
embeds the digest it attests near its start, which lets the hypothesis be checked directly
instead of reasoned about.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
OTS_MAGIC = b"\x00OpenTimestamps\x00\x00\x00"


def read_varint(b, i):
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


def embedded_digest(raw: bytes) -> bytes | None:
    """Skip the magic and the 'Proof' tag, then read the 32-byte attested digest."""
    if not raw.startswith(OTS_MAGIC):
        return None
    i = len(OTS_MAGIC)
    try:
        n, i = read_varint(raw, i)
        tag = raw[i:i + n]
        i += n
        if tag != b"Proof":
            return None
        return raw[i:i + 32]
    except Exception:
        return None


def main():
    recs = sorted((ID / "records").glob("*.pacsp"))
    print(f"  {'record':<46} {'extra hop?':>11} {'L4[:12]':>13} {'file[:12]':>13}")
    agree = 0
    checked = 0
    for p in recs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = rec.get("integrity", {}).get("L4", {}).get("data") or {}
        l4h = d.get("content_hash")
        proof = d.get("timestamp_proof")
        if not l4h or not proof:
            print(f"  {p.stem[:44]:<46} {'n/a':>11}")
            continue
        raw = base64.b64decode(proof.split(":", 1)[1]) if proof.startswith("base64:") \
            else bytes.fromhex(proof)
        emb = embedded_digest(raw)
        hyp = hashlib.sha256((l4h + "\n").encode("utf-8")).digest()
        ok = emb == hyp
        checked += 1
        agree += bool(ok)
        hf = Path(d.get("hash_file", "")) if d.get("hash_file") else None
        fname = hf.name[:12] if hf else "-"
        print(f"  {p.stem[:44]:<46} {str(ok):>11} {l4h[-12:]:>13} {fname:>13}")

    print(f"\n  hypothesis holds in {agree}/{checked} records")

    print("\n  chain, once the extra hop is known:")
    print("    corpus .txt files")
    print("      -> L1.content_hash            (stored in the record, and in L4.content_hash)")
    print("      -> sha256(content_hash + LF)  (what the calendars received; not stored)")
    print("      -> calendar PendingAttestation")
    print("      -> Bitcoin block header       (absent: never upgraded)")
    print("\n  so the binding is derivable but undocumented. A verifier holding the record")
    print("  can reconstruct the middle link, provided it knows the newline and the")
    print("  sha256: prefix were included in the stamped file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
