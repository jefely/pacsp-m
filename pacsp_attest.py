"""Attestation for pacsp records: L1 content hash, L2 signature, L3 Merkle, L4 timestamp.

This ports the four layers that PACSP-M's own results can support, so the tool can emit a
record that is tamper-evident and, when the ots client is present, anchored to Bitcoin.

What is deliberately not ported is L6, the innovation-dynamics layer. It depends on the
emotion-tree and IDL machinery that section 5.3 of the paper rejects, and a record should not
carry an attestation of a measure the same document shows to be unstable.

The layer contracts are modelled on PACSP-ID/scripts but are NOT wire-compatible with them,
and cross-verification fails in both directions. Measured:

    PACSP-ID verifier on a PACSP-M record  -> L1, L2, L3a, L3b, L3c all FAIL, L5 missing
    PACSP-M verifier on a PACSP-ID record  -> L1, L2, L3b, L3c FAIL, L4 ok

Three real differences explain it:

    L1   PACSP-ID folds an L6 reference into the root; PACSP-M folds an L3 reference, because
         L6 is not ported. Same idea, different digest.
    L2   PACSP-ID keeps the public key in an external file and verifies against that; PACSP-M
         carries the public key inside the record so a record is self-verifying.
    L5   PACSP-ID has an L5 reproducibility layer; PACSP-M does not produce one.

So a record is self-describing and carries its schema in metadata.record_schema
("pacsp-m/1"). A verifier should read that field and refuse to check a record whose schema it
does not implement, rather than reporting a mismatch as tampering. That is what this module's
verify_record does.

Two details carried over from the L4 audit, both of which were defects there:

  * `ots` attests a file, so what the calendars receive is the SHA-256 of the stamped file,
    not the content hash inside it. stamped_digest records that value, so the hop is visible.
  * `ots stamp` yields only a PendingAttestation. Completion needs `ots upgrade`, which is
    attempted here, and the block heights are read back out of the proof rather than taken
    from the command's exit status.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KEY_NAME = "pacsp_ed25519.key"
OTS_SUBDIR = "ots"
RECORD_SCHEMA = "pacsp-m/1"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def sha256_canonical(obj) -> str:
    """SHA-256 of the canonical JSON of obj. Matches pacsp_layer1._sha256_canonical."""
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def merkle_root(hashes: list[str]) -> str:
    """Binary Merkle root over hex digests, odd node promoted. Matches pacsp_merkle."""
    if not hashes:
        return "sha256:" + "0" * 64
    level = [h.split(":", 1)[1] if h.startswith("sha256:") else h for h in hashes]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(hashlib.sha256((a + b).encode()).hexdigest())
        level = nxt
    return "sha256:" + level[0]


# ---------------------------------------------------------------------------
# L1 / L3
# ---------------------------------------------------------------------------

def layer1(context: dict, l3: dict | None = None) -> dict:
    parts = [sha256_canonical(context.get(k) or {})
             for k in ("metadata", "dataset", "compute", "results",
                       "figure_recipe")]
    if l3:
        ref = sha256_canonical({"sample": l3.get("sample"),
                                "compute": l3.get("compute"),
                                "result": l3.get("result")})
    else:
        ref = "sha256:" + "0" * 64
    root = merkle_root(parts + [ref])
    return {"layer_id": "L1", "status": "ok",
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "data": {"content_hash": root, "hash_algorithm": "sha256",
                     "hash_scope": "layered",
                     "sub_hashes": dict(zip(
                         ("metadata", "dataset", "compute", "results",
                          "figure_recipe", "l3_reference"), parts + [ref]))},
            "error": None}


def layer3(context: dict, corpus_dir: Path | None) -> dict:
    """Three-level Merkle: the sample files, the compute outputs, the results."""
    sample = "sha256:" + "0" * 64
    n_files = 0
    if corpus_dir and Path(corpus_dir).is_dir():
        files = sorted(p for p in Path(corpus_dir).glob("*.txt"))
        n_files = len(files)
        if files:
            sample = merkle_root([sha256_file(f) for f in files])
    compute = merkle_root([sha256_canonical(context.get("compute") or {}),
                           sha256_canonical({"embeddings": "not-stored"})])
    result = merkle_root([sha256_canonical(context.get("results") or {}),
                          sha256_canonical({"figure_recipe":
                                            context.get("figure_recipe") or {}})])
    return {"layer_id": "L3", "status": "ok",
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "data": {"sample": sample, "compute": compute, "result": result,
                     "n_sample_files": n_files,
                     "levels": ["sample -> sha256 per file -> merkle",
                                "compute -> canonical json -> merkle",
                                "result -> canonical json -> merkle"]},
            "error": None}


# ---------------------------------------------------------------------------
# L2 signature
# ---------------------------------------------------------------------------

def _key_path(workdir: Path) -> Path:
    return Path(workdir) / KEY_NAME


def load_or_create_key(workdir: Path):
    """An Ed25519 key, created on first use and kept beside the records.

    The private key is written with owner-only permissions where the platform supports it.
    This is a local identity for tamper evidence, not a managed secret, and the function
    says so rather than implying more.
    """
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.hazmat.primitives import serialization
    p = _key_path(workdir)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        key = serialization.load_pem_private_key(p.read_bytes(), password=None)
        created = False
    else:
        key = ed25519.Ed25519PrivateKey.generate()
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption())
        p.write_bytes(pem)
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass
        created = True
    pub = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)
    return key, pub.hex(), created


def layer2(content_hash: str, workdir: Path) -> dict:
    key, pub_hex, created = load_or_create_key(workdir)
    sig = key.sign(content_hash.encode("utf-8"))
    return {"layer_id": "L2", "status": "ok",
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "data": {"algorithm": "Ed25519",
                     "public_key": pub_hex,
                     "public_key_id": pub_hex[:16],
                     "signature": base64.b64encode(sig).decode("ascii"),
                     "signed_payload": content_hash,
                     "key_created": created},
            "error": None}


def verify_layer2(record: dict, workdir: Path) -> tuple[bool, str]:
    from cryptography.hazmat.primitives.asymmetric import ed25519
    from cryptography.exceptions import InvalidSignature
    l2 = (record.get("integrity") or {}).get("L2") or {}
    d = l2.get("data") or {}
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(
            bytes.fromhex(d["public_key"]))
        pub.verify(base64.b64decode(d["signature"]),
                   d["signed_payload"].encode("utf-8"))
    except InvalidSignature:
        return False, "signature does not verify"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    l1 = (record.get("integrity") or {}).get("L1") or {}
    l1h = (l1.get("data") or {}).get("content_hash")
    if l1h != d.get("signed_payload"):
        return False, "signature is valid but covers a different content hash"
    return True, f"valid (key {d.get('public_key_id')})"


# ---------------------------------------------------------------------------
# L4 timestamp
# ---------------------------------------------------------------------------

def _ots_available() -> bool:
    return shutil.which("ots") is not None


def _ots_run(args: list[str], cache: Path, timeout: int = 120):
    """ots with --cache pointed into the workspace.

    The cache location is a command line option. Setting OTS_CACHE does nothing, which cost a
    full run during the L4 work: upgrade silently tried the user profile, hit a permission
    error and left every proof unchanged.
    """
    cmd = ["ots", "--cache", str(cache), *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as e:
        class R:
            returncode, stdout, stderr = 1, "", f"{type(e).__name__}: {e}"
        return R()


def _blocks_in(proof: Path) -> list[int]:
    r = _ots_run(["info", str(proof)], proof.parent)
    text = (r.stdout or "") + (r.stderr or "")
    out = []
    for m in re.finditer(r"BitcoinBlockHeaderAttestation\((\d+)\)", text):
        h = int(m.group(1))
        if h not in out:
            out.append(h)
    return out


def layer4(content_hash: str, workdir: Path, attempt_upgrade: bool = True) -> dict:
    ots_dir = Path(workdir) / OTS_SUBDIR
    ots_dir.mkdir(parents=True, exist_ok=True)
    hexpart = content_hash.split(":", 1)[1]
    stamped = ots_dir / f"{hexpart[:16]}.txt"
    stamped.write_text(content_hash + "\n", encoding="utf-8")
    stamped_digest = "sha256:" + hashlib.sha256(stamped.read_bytes()).hexdigest()

    base = {"content_hash": content_hash, "stamped_file": str(stamped),
            "stamped_digest": stamped_digest,
            "note_hop": "the calendars attest sha256(stamped_file), not content_hash"}

    if not _ots_available():
        return {"layer_id": "L4", "status": "pending",
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "data": {**base, "timestamp_proof": None,
                         "timestamp_anchor": None, "upgraded": False,
                         "bitcoin_attestations": [],
                         "note": "ots client not available; proof not created"},
                "error": None}

    r = _ots_run(["stamp", str(stamped)], ots_dir)
    proof = stamped.with_suffix(".txt.ots")
    if not proof.exists():
        return {"layer_id": "L4", "status": "failed",
                "computed_at": datetime.now(timezone.utc).isoformat(),
                "data": base,
                "error": f"stamp produced no proof: "
                         f"{(r.stderr or r.stdout or '')[:160]}"}

    blocks = _blocks_in(proof)
    upgraded = False
    if attempt_upgrade and not blocks:
        _ots_run(["upgrade", str(proof)], ots_dir, timeout=180)
        new = _blocks_in(proof)
        upgraded = bool(new)
        blocks = new

    proof_bytes = proof.read_bytes()
    status = "ok" if blocks else "pending"
    return {"layer_id": "L4", "status": status,
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "data": {**base,
                     "timestamp_proof":
                         "base64:" + base64.b64encode(proof_bytes).decode("ascii"),
                     "timestamp_anchor": ("bitcoin_block:" +
                                          ",".join(str(b) for b in blocks))
                     if blocks else "pending_bitcoin_confirmation",
                     "ots_file": str(proof),
                     "upgraded": upgraded,
                     "bitcoin_attestations": blocks,
                     "proof_bytes": len(proof_bytes),
                     "note": ("anchored in Bitcoin block(s) " +
                              ", ".join(str(b) for b in blocks)) if blocks else
                             "submitted; run again later, or ots upgrade, to complete"},
            "error": None}


# ---------------------------------------------------------------------------
# Record assembly and verification
# ---------------------------------------------------------------------------

def build_record(*, tool_version: str, frame: str, backend: str,
                 corpus_dirs: list[Path], stats: dict, workdir: Path,
                 attempt_upgrade: bool = True) -> dict:
    """Assemble a .pacsp record from a measurement.

    The context is what the layer functions hash, and it is shaped so the L1 and L3 digests
    actually cover the numbers reported: metadata names the frame and the corpus sizes,
    compute and results carry the statistics, and the sample level is the corpus files.
    """
    corpora = []
    for d in corpus_dirs:
        d = Path(d)
        files = sorted(d.glob("*.txt")) if d.is_dir() else []
        corpora.append({
            "name": d.name,
            "path_hint": d.name,
            "n_files": len(files),
            "files": [{"name": f.name, "bytes": f.stat().st_size,
                       "sha256": sha256_file(f).split(":", 1)[1]} for f in files],
        })

    context = {
        "metadata": {
            "tool": "pacsp",
            "tool_version": tool_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "frame": frame,
            "backend": backend,
            "record_schema": "pacsp-m/1",
        },
        "dataset": {"corpora": corpora,
                    "n_corpora": len(corpora),
                    "total_files": sum(c["n_files"] for c in corpora)},
        "compute": {"method": "mean pairwise embedding distance",
                    "frame": frame,
                    "backend": backend,
                    "embeddings_stored": False,
                    "note": "embeddings are not stored; recomputable from the corpus "
                            "and the declared frame"},
        "results": stats,
        "figure_recipe": {"kind": "none"},
    }

    l3 = layer3(context, corpus_dirs[0] if corpus_dirs else None)
    l1 = layer1(context, l3.get("data"))
    content_hash = l1["data"]["content_hash"]
    l2 = layer2(content_hash, workdir)
    l4 = layer4(content_hash, workdir, attempt_upgrade=attempt_upgrade)

    return {
        "version": "1.0.0-PACSP-M",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": context["metadata"],
        "dataset": context["dataset"],
        "compute": context["compute"],
        "results": context["results"],
        "figure_recipe": context["figure_recipe"],
        "integrity": {"L1": l1, "L2": l2, "L3": l3, "L4": l4},
    }


def upgrade_proof(record: dict, workdir: Path) -> tuple[bool, list[int], str]:
    """Ask the calendars whether a pending proof can be completed now.

    A pending proof means the calendars accepted the submission but had not yet committed it
    to a block when the record was written. That takes one to six hours, so a user who wants
    the anchor has to come back. This is what they call to do it, and it reports what the
    proof contains afterwards rather than what the command claimed.

    Returns (changed, block_heights, message).
    """
    l4d = ((record.get("integrity") or {}).get("L4") or {}).get("data") or {}
    proof = Path(l4d.get("ots_file") or "")
    if not proof.exists():
        alt = (Path(workdir) / OTS_SUBDIR /
               f"{(l4d.get('content_hash') or 'x').split(':', 1)[-1][:16]}.txt.ots")
        if alt.exists():
            proof = alt
        else:
            return False, [], f"proof file not found ({proof or alt})"
    if not _ots_available():
        return False, [], "ots client not available"

    before = _blocks_in(proof)
    if before:
        return False, before, "already anchored"
    _ots_run(["upgrade", str(proof)], proof.parent.parent, timeout=240)
    after = _blocks_in(proof)
    if after:
        l4d["bitcoin_attestations"] = after
        l4d["timestamp_anchor"] = "bitcoin_block:" + ",".join(str(b) for b in after)
        l4d["upgraded"] = True
        l4d["timestamp_proof"] = ("base64:" +
                                 base64.b64encode(proof.read_bytes()).decode("ascii"))
        l4d["proof_bytes"] = proof.stat().st_size
        l4d["note"] = "anchored in Bitcoin block(s) " + ", ".join(str(b) for b in after)
        record["integrity"]["L4"]["data"] = l4d
        record["integrity"]["L4"]["status"] = "ok"
        return True, after, "upgraded"
    return False, [], "still pending; the calendars have not committed it yet"


def verify_record(record: dict, corpus_dir: Path | None = None,
                  workdir: Path | None = None) -> dict:
    """Recompute what can be recomputed and report each layer separately.

    L1 is recomputed from the stored context, so any edit to the numbers breaks it. L2 is
    checked against the key in the record. L3's sample level is rechecked when the corpus is
    supplied. L4 is reported from what the proof contains, not from the anchor string, since
    the failure being guarded against is a record claiming an anchor it does not hold.

    A record whose schema is not pacsp-m/1 is reported as such instead of being checked and
    called broken. The two formats hash different things at L1, so a foreign record would
    fail on every layer and look tampered when it is merely different.
    """
    out = {}
    schema = ((record.get("metadata") or {}).get("record_schema")
              or record.get("version"))
    if schema != RECORD_SCHEMA:
        return {"_schema": (None, f"record schema {schema!r} is not {RECORD_SCHEMA}; "
                                  "this verifier implements pacsp-m/1 only and will not "
                                  "report a mismatch as tampering")}
    integ = record.get("integrity") or {}

    l3 = integ.get("L3") or {}
    recomputed_l1 = layer1({k: record.get(k) for k in
                            ("metadata", "dataset", "compute", "results",
                             "figure_recipe")}, (l3.get("data") or {}))
    stored_l1 = ((integ.get("L1") or {}).get("data") or {}).get("content_hash")
    out["L1"] = (stored_l1 == recomputed_l1["data"]["content_hash"],
                 f"stored {str(stored_l1)[:22]} recomputed "
                 f"{recomputed_l1['data']['content_hash'][:22]}")

    if integ.get("L2"):
        out["L2"] = verify_layer2(record, workdir or Path("."))
    else:
        out["L2"] = (False, "absent")

    if corpus_dir and (l3.get("data") or {}).get("sample"):
        files = sorted(Path(corpus_dir).glob("*.txt"))
        got = merkle_root([sha256_file(f) for f in files]) if files else None
        want = l3["data"]["sample"]
        out["L3a"] = (got == want,
                      f"{len(files)} files, stored {str(want)[:22]}")
    else:
        out["L3a"] = (None, "corpus not supplied; sample level not rechecked")

    l3c = (l3.get("data") or {}).get("compute")
    recomputed_c = layer3({k: record.get(k) for k in
                           ("compute", "results", "figure_recipe")}, None)
    out["L3b"] = (l3c == recomputed_c["data"]["compute"],
                  "compute level recomputed" if l3c else "absent")
    out["L3c"] = ((l3.get("data") or {}).get("result") ==
                  recomputed_c["data"]["result"],
                  "result level recomputed")

    l4d = (integ.get("L4") or {}).get("data") or {}
    blocks = l4d.get("bitcoin_attestations") or []
    anchor = l4d.get("timestamp_anchor") or "none"
    out["L4"] = (bool(blocks) if blocks else None,
                 f"anchor {anchor}" +
                 (f", {len(blocks)} block(s)" if blocks else
                  "; pending, not an anchor"))
    return out
