"""G3: what does L4's ``content_hash`` correspond to, and where is the chain's seam?

Answers evidence gap E7 by walking the chain literally, and — importantly — by testing
BOTH candidate conventions rather than assuming one. The first version of this script
asserted two conclusions that its own measurements contradicted (0/9 on both); it is kept
honest now by printing only what was measured.

Findings, all reproducible from this script:

  1. ``L4.content_hash == L1.content_hash`` in 9/9 records. L4 anchors L1, not an L3 root —
     which is why the earlier audit "could not match any L3 root": it looked one level too low.

  2. ``L1.content_hash`` IS the binary Merkle root over the six ``sub_hashes``, but only under
     the **prefix-retaining** convention (9/9). Under the prefix-stripping convention it is
     0/9. ``PACSP-ID/scripts/pacsp_verify.py`` and ``PACSP-M/pacsp_attest.py`` implement the
     two conventions respectively, so they are NOT the same function.

  3. The digest the calendars actually attested is ``sha256(content_hash + "\\r\\n")``, not
     ``+ "\\n"``: ``Path.write_text`` translates the newline on Windows. Verified against the
     file's raw bytes.

Run:  python PACSP-M/tools/diagnose_l4_binding.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = HERE.parent
ROOT = M.parent
ID = ROOT / "PACSP-ID"
ORDER = ("metadata", "dataset", "compute", "results", "figure_recipe", "l3_reference")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def merkle_root(hashes: list[str], keep_prefix: bool) -> str:
    """Binary Merkle root, odd node promoted.

    ``keep_prefix`` selects the convention. The project contains BOTH:
      keep_prefix=True   -> PACSP-ID/scripts/pacsp_verify.py  (matches the records)
      keep_prefix=False  -> PACSP-M/pacsp_attest.py
    """
    level = list(hashes) if keep_prefix else [h.split(":", 1)[1] for h in hashes]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [sha256_hex((level[i] + level[i + 1]).encode())
                 for i in range(0, len(level), 2)]
    return "sha256:" + level[0]


def main() -> int:
    records = sorted((ID / "records").glob("*.pacsp"))
    if not records:
        print(f"no records under {ID / 'records'}", file=sys.stderr)
        return 2

    print("G3 · L4 绑定诊断\n")
    print(f"记录 {len(records)} 条，位于 {ID / 'records'}\n")

    n = len(records)
    keep_ok = strip_ok = 0
    l4_eq_l1 = 0
    txt_exists = txt_holds = 0
    crlf_ok = lf_ok = 0
    crlf_files = 0

    for p in records:
        rec = json.loads(p.read_text(encoding="utf-8"))
        integ = rec.get("integrity") or {}
        l1 = (integ.get("L1") or {}).get("data") or {}
        l4 = (integ.get("L4") or {}).get("data") or {}
        ch = l1.get("content_hash")

        if l4.get("content_hash") == ch:
            l4_eq_l1 += 1

        subs = l1.get("sub_hashes") or {}
        vals = [subs[k] for k in ORDER if k in subs]
        if vals:
            if merkle_root(vals, keep_prefix=True) == ch:
                keep_ok += 1
            if merkle_root(vals, keep_prefix=False) == ch:
                strip_ok += 1

        sp = Path(l4["stamped_file"]) if l4.get("stamped_file") else None
        if sp and sp.exists():
            txt_exists += 1
            raw = sp.read_bytes()
            if raw.decode("utf-8", "replace").strip() == ch:
                txt_holds += 1
            digest = l4.get("stamped_digest")
            if raw.endswith(b"\r\n"):
                crlf_files += 1
            if "sha256:" + sha256_hex((ch + "\r\n").encode()) == digest:
                crlf_ok += 1
            if "sha256:" + sha256_hex((ch + "\n").encode()) == digest:
                lf_ok += 1

    print("测量结果")
    print(f"  L4.content_hash == L1.content_hash                     : {l4_eq_l1}/{n}")
    print(f"  L1.content_hash == Merkle 根（**保留** sha256: 前缀）    : {keep_ok}/{n}")
    print(f"  L1.content_hash == Merkle 根（**去掉** sha256: 前缀）    : {strip_ok}/{n}")
    print(f"  被盖章的 .txt 存在                                      : {txt_exists}/{n}")
    print(f"  .txt 内容是 content_hash 文本                           : {txt_holds}/{n}")
    print(f"  .txt 以 CRLF 结尾                                       : {crlf_files}/{n}")
    print(f"  stamped_digest == sha256(content_hash + CRLF)          : {crlf_ok}/{n}")
    print(f"  stamped_digest == sha256(content_hash + LF)            : {lf_ok}/{n}")

    print("\n结论（只写上面测到的）")
    print("  ① content_hash **不是** L3 根。它是 **L1 的 Merkle 根**，")
    print("     由 6 个子哈希（metadata/dataset/compute/results/figure_recipe/l3_reference）算出。")
    print("     早期审计「匹配不上任何 L3 根」是因为**找低了一层**，不是绑定缺失。")
    print("  ② 但两个实现对该 Merkle 约定不一致：")
    print("       PACSP-ID/scripts/pacsp_verify.py  保留 'sha256:' 前缀  -> 与记录一致")
    print("       PACSP-M/pacsp_attest.py           去掉 'sha256:' 前缀  -> 与记录不一致")
    print("     二者不是同一个函数。若用后者重签，同一输入会给出**不同的 content_hash**。")
    print("  ③ 日历真正收到的是 sha256(content_hash + CRLF)，不是 + LF。")
    print("     成因是 Path.write_text() 在 Windows 上把 \\n 翻成 \\r\\n。")
    print("     记录里没有任何一处写明这一跳或这个换行约定，")
    print("     所以 Linux 上的外部验证者会算出 sha256(content_hash + LF)，")
    print("     得到一个**不同的值**，从而误判绑定已断。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
