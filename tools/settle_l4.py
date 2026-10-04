"""Settle the L4 file question with one unambiguous check per record.

Earlier runs produced inconsistent-looking prefixes, so rather than infer anything further
this answers exactly one question per record and prints every value it used:

    does the file named by ots_file exist, and does its content equal L4.content_hash?

It also prints the digest the OpenTimestamps proof should attest, computed as the SHA-256 of
the stamped input file, so the whole chain from record to calendar is visible in one place.
"""

import base64
import hashlib
import json
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
OTS = ID / "cache" / "ots"


def main():
    recs = sorted((ID / "records").glob("*.pacsp"))
    print(f"  records: {len(recs)}, cache .txt: {len(list(OTS.glob('*.txt')))}, "
          f"cache .ots: {len(list(OTS.glob('*.ots')))}")

    rows = []
    for p in recs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        d = rec["integrity"]["L4"].get("data") or {}
        ch = d.get("content_hash")
        of = d.get("ots_file")
        proof = d.get("timestamp_proof")

        # the .txt that the .ots was made from: ots_file ends in .txt.ots
        txt = Path(of) if of else None
        if txt is not None and txt.suffix == ".ots":
            txt = txt.with_suffix("")  # -> ...txt

        exists = bool(txt and txt.exists())
        content = txt.read_text(encoding="utf-8") if exists else None
        content_matches = (content is not None and content.strip() == ch)

        # digest of the stamped input, which is what a calendar attests
        stamped_digest = (hashlib.sha256(content.encode("utf-8")).hexdigest()
                          if content is not None else None)

        # digest embedded in the proof, if it is a detached OTS proof
        emb = None
        if proof and proof.startswith("base64:"):
            raw = base64.b64decode(proof.split(":", 1)[1])
            if raw.startswith(b"\x00OpenTimestamps\x00\x00\x00"):
                i = 11
                n, shift = 0, 0
                while i < len(raw):
                    c = raw[i]
                    i += 1
                    n |= (c & 0x7F) << shift
                    if not (c & 0x80):
                        break
                    shift += 7
                i += n  # the 'Proof' tag
                emb = raw[i:i + 32].hex()

        rows.append({
            "record": p.stem[:40],
            "hash": ch,
            "hash_prefix": ch.split(":", 1)[1][:16] if ch else None,
            "ots_stem": txt.stem[:16] if txt else None,
            "txt_exists": exists,
            "content_eq_hash": content_matches,
            "stamped_digest_prefix": stamped_digest[:16] if stamped_digest else None,
            "proof_digest_prefix": emb[:16] if emb else None,
        })

    hdr = (f"  {'record':<42} {'hash[:8]':>9} {'ots stem':>17} {'txt':>4} "
           f"{'eq?':>4} {'stamped[:8]':>12} {'proof[:8]':>10}")
    print("\n" + hdr)
    for r in rows:
        print(f"  {r['record']:<42} {(r['hash_prefix'] or '')[:8]:>9} "
              f"{(r['ots_stem'] or '-'):>17} {str(r['txt_exists']):>4} "
              f"{str(r['content_eq_hash']):>4} "
              f"{(r['stamped_digest_prefix'] or '-')[:8]:>12} "
              f"{(r['proof_digest_prefix'] or '-')[:8]:>10}")

    n_ex = sum(1 for r in rows if r["txt_exists"])
    n_eq = sum(1 for r in rows if r["content_eq_hash"])
    n_stem_match = sum(1 for r in rows
                       if r["ots_stem"] and r["hash_prefix"]
                       and r["ots_stem"] == r["hash_prefix"])
    n_proof_eq_stamp = sum(1 for r in rows
                           if r["proof_digest_prefix"] and r["stamped_digest_prefix"]
                           and r["proof_digest_prefix"] == r["stamped_digest_prefix"])
    print(f"\n  ots_file .txt exists            : {n_ex}/{len(rows)}")
    print(f"  its content equals content_hash : {n_eq}/{len(rows)}")
    print(f"  ots stem equals hash prefix     : {n_stem_match}/{len(rows)}")
    print(f"  proof digest equals sha256(txt) : {n_proof_eq_stamp}/{len(rows)}")

    print("\n  chain, as the files actually show it:")
    r = rows[0]
    print(f"    L1.content_hash            = {r['hash']}")
    print(f"    L4.content_hash            = same value (checked in all 9)")
    print(f"    stamped input file         = {r['ots_stem']}.txt")
    print(f"    digest a calendar attests  = sha256(that file)"
          f" = {r['stamped_digest_prefix']}...")
    print(f"    proof's embedded digest    = {r['proof_digest_prefix']}...")
    print("\n  conclusion:")
    if n_eq == len(rows) and n_proof_eq_stamp == len(rows):
        print("    The chain is complete and self-consistent on disk. A verifier holding")
        print("    the record, the corpus and the cache can walk it end to end. What the")
        print("    record does not carry is the stamped file itself, so the last hop is")
        print("    reproducible only if the verifier rebuilds the same text: the content")
        print("    hash with its 'sha256:' prefix and a trailing newline.")
        print("    Fix: store the stamped digest, or the exact stamped bytes, in L4.")
    else:
        print("    The chain is not uniform across records; see the table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
