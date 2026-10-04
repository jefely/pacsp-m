"""Attestation end-to-end: build a record, verify it, break it, and check the schema guard.

Four things are asserted, and the fourth matters as much as the first three.

    round trip   a record built by the tool verifies, including the sample Merkle level
    tampering    editing a reported number breaks L1 and L3c
    signature    L2 verifies against the key the record carries, with no external key file
    schema       a record from another format is refused rather than reported as tampered

The last one exists because cross-verification with PACSP-ID fails in both directions for
legitimate reasons: the two formats fold different things into L1 and PACSP-ID keeps its
public key outside the record. A verifier that reported those as tampering would be worse than
one that declined to check.

Not asserted: that L4 reaches a Bitcoin anchor. The calendars take one to six hours to commit a
submission, so a test cannot wait for it. The record is checked to say "pending" honestly and to
never claim a block it does not have.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

M = Path(r"D:\myproject\PACSP-M")
ID = Path(r"D:\myproject\PACSP-ID")
sys.path.insert(0, str(M))

import pacsp_attest  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok


def main():
    work = Path(tempfile.mkdtemp(prefix="pacsp_attest_"))
    corpus = M / "data" / "poem"
    print(f"  workdir: {work}")
    print(f"  corpus : {corpus}  ({len(list(corpus.glob('*.txt')))} files)")

    # ---------------------------------------------------------------- build
    print("\n=== build a record ===")
    stats = {"kind": "measure", "collection": "poem", "n": 31, "D": 0.543351,
             "ci95": [0.526, 0.561], "bootstrap_cv": 0.0165, "n_distances": 465}
    rec = pacsp_attest.build_record(
        tool_version="1.1.0", frame="bge-large-zh", backend="onnx",
        corpus_dirs=[corpus], stats=stats, workdir=work, attempt_upgrade=False)
    path = work / "poem.pacsp"
    path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  wrote {path}  ({path.stat().st_size} bytes)")
    for k in ("L1", "L2", "L3", "L4"):
        print(f"    {k}: {rec['integrity'][k]['status']}")
    check("record carries the pacsp-m/1 schema",
          rec["metadata"]["record_schema"] == "pacsp-m/1")
    check("L1, L2, L3 all ok",
          all(rec["integrity"][k]["status"] == "ok" for k in ("L1", "L2", "L3")))

    # ---------------------------------------------------------------- verify
    print("\n=== verify it ===")
    res = pacsp_attest.verify_record(rec, corpus, work)
    for k in ("L1", "L2", "L3a", "L3b", "L3c", "L4"):
        ok, msg = res.get(k, (None, "missing"))
        print(f"    {k:<4} {str(ok):<6} {msg}")
    check("L1 recomputes to the stored hash", res["L1"][0] is True, res["L1"][1][:60])
    check("L2 signature verifies from the record alone", res["L2"][0] is True,
          res["L2"][1])
    check("L3a sample Merkle matches the corpus", res["L3a"][0] is True)
    check("L3b compute level recomputes", res["L3b"][0] is True)
    check("L3c result level recomputes", res["L3c"][0] is True)

    # ---------------------------------------------------------------- sample hash cross-check
    print("\n=== the sample level matches PACSP-ID's published value for poem ===")
    got = rec["integrity"]["L3"]["data"]["sample"]
    print(f"    computed: {got}")
    print(f"    PACSP-ID: sha256:3b1d11ddbc85a29...")
    check("sample Merkle root agrees with the frozen PACSP-ID record",
          got.startswith("sha256:3b1d11ddbc85a29"), got)

    # ---------------------------------------------------------------- tamper
    print("\n=== tamper with a reported number ===")
    for field, label in ((("D",), "D"), (("n",), "n")):
        bad = json.loads(json.dumps(rec))
        bad["results"][field[0]] = 999.99 if field[0] == "D" else 7
        r = pacsp_attest.verify_record(bad, corpus, work)
        check(f"editing results.{label} breaks L1", r["L1"][0] is False)
        check(f"editing results.{label} breaks L3c", r["L3c"][0] is False)

    bad2 = json.loads(json.dumps(rec))
    bad2["dataset"]["corpora"][0]["n_files"] = 99
    r2 = pacsp_attest.verify_record(bad2, corpus, work)
    check("editing the dataset block breaks L1", r2["L1"][0] is False)

    bad3 = json.loads(json.dumps(rec))
    bad3["integrity"]["L2"]["data"]["signature"] = "AAAA"
    r3 = pacsp_attest.verify_record(bad3, corpus, work)
    check("a corrupted signature fails L2", r3["L2"][0] is False, r3["L2"][1][:60])

    # ---------------------------------------------------------------- wrong corpus
    print("\n=== verifying against the wrong corpus ===")
    r4 = pacsp_attest.verify_record(rec, M / "data" / "lyrics", work)
    check("L3a fails when the corpus does not match", r4["L3a"][0] is False,
          r4["L3a"][1][:60])

    # ---------------------------------------------------------------- schema guard
    print("\n=== a record from another format is refused, not called tampered ===")
    foreign = ID / "records" / "poem_epoch1_base_CT1.90Se_20261004.pacsp"
    if foreign.exists():
        frec = json.loads(foreign.read_text(encoding="utf-8"))
        r5 = pacsp_attest.verify_record(frec, ID / "data" / "poem", work)
        check("foreign schema yields a skip, not failures",
              "_schema" in r5 and len(r5) == 1, str(r5.get("_schema", ("", ""))[1])[:70])
    else:
        print("  (PACSP-ID record not present; skipped)")

    # ---------------------------------------------------------------- L4 honesty
    print("\n=== L4 does not claim an anchor it lacks ===")
    l4d = rec["integrity"]["L4"]["data"]
    check("with no ots attempt, status is pending",
          rec["integrity"]["L4"]["status"] == "pending")
    check("no block heights are claimed",
          not l4d.get("bitcoin_attestations"), str(l4d.get("bitcoin_attestations")))
    check("the stamped digest is recorded, distinct from the content hash",
          l4d.get("stamped_digest") and
          l4d["stamped_digest"] != l4d["content_hash"])

    if pacsp_attest._ots_available():
        print("\n=== with the ots client present, a real proof is produced ===")
        rec2 = pacsp_attest.build_record(
            tool_version="1.1.0", frame="bge-large-zh", backend="onnx",
            corpus_dirs=[corpus], stats=stats, workdir=work, attempt_upgrade=False)
        l4 = rec2["integrity"]["L4"]
        check("a proof file exists", bool(l4["data"].get("ots_file")) and
              Path(l4["data"]["ots_file"]).exists(),
              f"{l4['data'].get('proof_bytes')} bytes")
        check("anchor is either a block or honestly pending",
              l4["data"]["timestamp_anchor"].startswith("bitcoin_block:")
              or l4["data"]["timestamp_anchor"] == "pending_bitcoin_confirmation",
              l4["data"]["timestamp_anchor"])
    else:
        print("\n  (ots client absent; proof creation not exercised)")

    print()
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED:")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  attestation verified end to end")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
