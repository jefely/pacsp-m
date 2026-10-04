"""Upgrade every pending L4 proof and write the Bitcoin anchor back into its record.

The 2026-10-04 records carry PendingAttestations. The calendars had already anchored them;
what never happened was the upgrade step, so the proofs sat at 840 bytes with no block header
path while the chain held the anchor all along. Upgrading turns them into complete proofs.

Two things this does carefully.

It does not overwrite a record in place without a backup, and it does not claim an anchor it
has not read back. The block heights written into the record are parsed from `ots info` on the
upgraded proof, not taken from the upgrade command's exit status, because the failure mode
being guarded against is a record asserting something its proof does not contain.

It also records what was actually stamped. The digest the calendars received is the SHA-256 of
the stamped file, not the content hash stored in the record, and the record previously said
nothing about that hop, which is why the binding could not be recovered from the record alone.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ID = Path(r"D:\myproject\PACSP-ID")
RECORDS = ID / "records"
OTS_DIR = ID / "cache" / "ots"
CACHE = OTS_DIR / "cache"
STAMP = "20261005"


def ots(args, timeout=180):
    """Run ots with --cache pointed into the workspace.

    The client's cache location is a command line option, not an environment variable. An
    earlier version of this script set OTS_CACHE, which the client ignores, so upgrade kept
    trying to write under the user profile, failed with a permission error, and left every
    proof unchanged. The --cache flag is what the working manual run used.
    """
    cmd = ["ots", "--cache", str(CACHE), *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout)
    except Exception as e:
        class R:
            returncode, stdout, stderr = 1, "", f"{type(e).__name__}: {e}"
        return R()


def blocks_in(ots_file: Path) -> list[int]:
    r = ots(["info", str(ots_file)])
    text = (r.stdout or "") + (r.stderr or "")
    out = []
    for m in re.finditer(r"BitcoinBlockHeaderAttestation\((\d+)\)", text):
        h = int(m.group(1))
        if h not in out:
            out.append(h)
    return out


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    recs = sorted(RECORDS.glob("*.pacsp"))
    print(f"  records: {len(recs)}")
    print(f"  cache  : {CACHE}")

    rows = []
    for rec_path in recs:
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        l4 = rec.get("integrity", {}).get("L4", {})
        data = l4.get("data") or {}
        ch = data.get("content_hash")
        ots_path = Path(data.get("ots_file") or "")
        if not ch or not ots_path.name:
            print(f"  {rec_path.stem[:46]:<48} skip (no content_hash or ots_file)")
            continue
        if not ots_path.exists():
            # fall back to the naming convention
            alt = OTS_DIR / f"{ch.split(':', 1)[1][:16]}.txt.ots"
            ots_path = alt if alt.exists() else ots_path
        if not ots_path.exists():
            print(f"  {rec_path.stem[:46]:<48} skip (proof missing)")
            continue

        before = ots_path.stat().st_size
        had = blocks_in(ots_path)
        changed = False
        upgrade_note = ""
        if not had:
            r = ots(["upgrade", str(ots_path)])
            had = blocks_in(ots_path)
            changed = bool(had)
            if not had:
                # surface why, instead of silently reporting pending
                tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
                upgrade_note = tail[-1][:90] if tail else f"rc={r.returncode}"
        after = ots_path.stat().st_size

        # the stamped file and the digest the calendars actually received
        stamped_file = ID / "cache" / "ots" / f"{ch.split(':', 1)[1][:16]}.txt"
        stamped_digest = None
        if stamped_file.exists():
            stamped_digest = "sha256:" + hashlib.sha256(
                stamped_file.read_bytes()).hexdigest()

        # only claim what the proof now contains
        status = "ok" if had else l4.get("status", "pending")
        data.update({
            "content_hash": ch,
            "stamped_file": str(stamped_file) if stamped_file.exists() else None,
            "stamped_digest": stamped_digest,
            "ots_file": str(ots_path),
            "upgraded": changed or bool(had),
            "bitcoin_attestations": had,
            "timestamp_anchor": ("bitcoin_block:" + ",".join(str(b) for b in had)
                                 if had else data.get("timestamp_anchor")),
            "note": ("已上链；区块 " + ", ".join(str(b) for b in had)) if had
                    else data.get("note"),
        })
        data["timestamp_proof"] = ("base64:" + base64.b64encode(
            ots_path.read_bytes()).decode("ascii"))
        l4["data"] = data
        l4["status"] = status
        rec["integrity"]["L4"] = l4

        backup = rec_path.with_suffix(f".pacsp.pre-{STAMP}")
        if not backup.exists():
            shutil.copy2(rec_path, backup)
        rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        rows.append({"record": rec_path.stem, "blocks": had,
                     "before_bytes": before, "after_bytes": after,
                     "stamped_digest": stamped_digest,
                     "upgrade_note": upgrade_note})
        print(f"  {rec_path.stem[:46]:<48} {before:>5}->{after:<5} B  "
              f"blocks {had if had else 'pending'}"
              + (f"   [{upgrade_note}]" if upgrade_note else ""))

    print(f"\n=== summary ===")
    anchored = [r for r in rows if r["blocks"]]
    print(f"  records upgraded with a Bitcoin anchor: {len(anchored)}/{len(rows)}")
    allb = sorted({b for r in anchored for b in r["blocks"]})
    print(f"  distinct Bitcoin block heights        : {allb}")
    print(f"  backups written as *.pacsp.pre-{STAMP}")

    (ID / "records_centroid").mkdir(exist_ok=True)
    (ID / "records_centroid" / f"l4_upgrade_{STAMP}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written records_centroid/l4_upgrade_{STAMP}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
