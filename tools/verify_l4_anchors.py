"""Confirm the upgraded records verify, and check the Bitcoin blocks independently.

Two different questions.

First, did rewriting L4 break anything? pacsp_verify checks L1, L2, L3a, L3b, L3c and L5, and
L4 is non-fatal, so a broken L4 would still pass. That makes it worth checking L4 explicitly
rather than relying on the overall verdict.

Second, are the block heights real? A block height written into a record proves nothing on its
own. This asks a public Bitcoin API for each height and reports the timestamp and hash, which
is independent of anything this repository produced. Until that agrees, the anchors are
assertions rather than evidence.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
M = Path(r"D:\myproject\PACSP-M")
sys.path.insert(0, str(ID / "scripts"))

BLOCKS = [967305, 967320, 967330, 967812, 969821, 969867, 969869]
APIS = [
    "https://blockstream.info/api/block-height/{h}",
    "https://mempool.space/api/block-height/{h}",
]


def get(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "pacsp-verify"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace").strip()


def main():
    print("=== 1. records still verify after the L4 rewrite ===")
    import pacsp_verify as V
    ok_all = True
    for p in sorted((ID / "records").glob("*.pacsp")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        l1 = V.verify_l1(rec)
        l2 = V.verify_l2(rec)
        l4 = V.verify_l4(rec)
        blocks = (rec.get("integrity", {}).get("L4", {}).get("data") or {}).get(
            "bitcoin_attestations") or []
        good = l1[0] is True and l2[0] is True and l4[0] is True and bool(blocks)
        ok_all = ok_all and good
        print(f"  [{'ok  ' if good else 'FAIL'}] {p.stem[:44]:<46} "
              f"L1={l1[0]} L2={l2[0]} L4={l4[0]} blocks={len(blocks)}")
    print(f"  all records carry an anchor and pass L1/L2/L4: {ok_all}")

    print("\n=== 2. the Bitcoin blocks, from a public API ===")
    print(f"  {'height':>8} {'hash (first 20)':<22} {'mined (UTC)':<22} source")
    seen = {}
    for h in BLOCKS:
        got = None
        for tpl in APIS:
            try:
                hh = get(tpl.format(h=h))
                got = (hh, tpl.split("/")[2])
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                continue
        if not got:
            print(f"  {h:>8} {'':<22} {'API unreachable':<22}")
            continue
        hh, src = got
        # the hash endpoint returns the block hash; the hash endpoint by height is
        # available from both services
        try:
            raw = get(f"https://blockstream.info/api/block/{hh}")
            b = json.loads(raw)
            import datetime
            ts = datetime.datetime.utcfromtimestamp(
                b["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = "?"
        seen[h] = {"hash": hh, "time": ts, "source": src}
        print(f"  {h:>8} {hh[:20]:<22} {ts:<22} {src}")

    print("\n=== 3. does the anchored time bracket the submission? ===")
    print("  the records were generated 2026-10-04 (UTC), per their computed_at fields")
    for h, info in sorted(seen.items()):
        print(f"    block {h}: {info['time']} UTC")

    print("\n=== 4. the stamped digest is derivable from the record ===")
    p = ID / "records" / "human_poem_epoch1_base_CT11.54Se_20261004.pacsp"
    rec = json.loads(p.read_text(encoding="utf-8"))
    d = rec["integrity"]["L4"]["data"]
    ch = d["content_hash"]
    stamped_file = Path(d["stamped_file"]) if d.get("stamped_file") else None
    print(f"    content_hash   : {ch}")
    if stamped_file and stamped_file.exists():
        raw = stamped_file.read_bytes()
        derived = "sha256:" + hashlib.sha256(raw).hexdigest()
        print(f"    stamped file   : {stamped_file.name}")
        print(f"    sha256 of it   : {derived}")
        print(f"    recorded digest: {d['stamped_digest']}")
        print(f"    agree          : {derived == d['stamped_digest']}")
    else:
        print("    stamped file missing; the record names it but the cache does not have it")
        print(f"    recorded digest: {d.get('stamped_digest')}")

    out = {"records_ok": ok_all, "blocks": seen}
    (M / "records_centroid" / "l4_independent_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {M / 'records_centroid' / 'l4_independent_check.json'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
