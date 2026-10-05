"""Assemble PACSP-M 1.2.0 from 1.1.0's retained sections plus the new ones.

WHY A SCRIPT RATHER THAN RETYPING
---------------------------------
1.1.0 is 1490 lines. Sections 1, 2, 4, the veto section, the reporting rules, the
non-claims and appendices A/B are genuine assets that must survive *verbatim* — retyping
them would introduce transcription errors into a document whose whole argument is that
claims should be traceable. So the kept sections are copied byte-for-byte out of 1.1.0,
and only the sections that actually change are authored fresh in `paper-1.2.0-src/`.

WHAT IT DOES
------------
  * splits 1.1.0 on `## ` headings
  * keeps the named sections, RENUMBERING their headings and their sub-headings
  * remaps the old `§N` cross-references to the new numbers
  * splices in the newly authored sections at the right positions
  * VERIFIES: every `§N` reference in the result resolves to a heading that exists,
    and reports the byte count kept vs authored

The verification is the point. A renumbering that silently breaks cross-references would
be exactly the kind of unverified edit this project keeps finding in its own history.

Run:  python tools/build_paper_1_2.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parent.parent
SRC = M / "paper-1.2.0-src"
OLD = M / "PACSP-M-1.1.0.md"
NEW = M / "PACSP-M-1.2.0.md"

# 1.1.0 section title -> new number, for the sections carried over unchanged
RENUMBER = {
    "1. 框架的构造原则": 1,
    "2. 参照系": 2,
    "4. 核心度量": 4,
    "5. 被实测否决的部分": 9,
    "6. 报告规范": 10,
    "7. 框架明确不主张的": 11,
    "附录 A：可复现性": None,      # appendices keep their letter
    "附录 B：本文引用的全部数值及其来源": None,
    "参考文献": None,
}

# old section number -> new, for `§N` / `§N.x` cross-references
REF_REMAP = {"§5": "§9", "§6": "§10", "§7": "§11", "§8": "§12", "§3.4": "§3.3"}

# Explicit edits applied to KEPT sections. Each anchor must occur EXACTLY ONCE, or the
# build fails — a patch that silently does not apply is exactly the kind of unverified
# edit this project keeps catching in its own history.
PATCHES: dict[str, list[tuple[str, str]]] = {
    "6. 报告规范": [
        (
            "| 5 | **换一条独立路径复核** | 把单一方法伪迹当现象 | 自由生成 vs 受限读数方向相反 |",
            "| 5 | **换一条独立路径复核** | 把单一方法伪迹当现象 | 自由生成 vs 受限读数方向相反 |\n"
            "| 6 | **报告顺序敏感度与方向保持率** | 把承载信号的性质当缺陷消除 | "
            "§3.3.1：全部 `μ` 变体重排 200 次，方向保持 **100%** |",
        ),
        (
            "方向一致性  : <k>/<m> 个配合同向\n例外        : <列出反向的配合及其区间>",
            "方向一致性  : <k>/<m> 个配合同向\n例外        : <列出反向的配合及其区间>\n"
            "顺序敏感度  : as-filed <值> / 重排均值 <值> / 方向保持率 <k>/<m>\n"
            "              （用于序列表示；见 §3.3.1——只报数值不报方向保持率会误判）",
        ),
    ],
}

# assembly order: source file (in SRC) or ("keep", "1.1.0 section title")
ASSEMBLY = [
    ("file", "00-frontmatter.md"),
    ("file", "01-abstract.md"),
    ("keep", "1. 框架的构造原则"),
    ("keep", "2. 参照系"),
    ("file", "03-representation.md"),
    ("keep", "4. 核心度量"),
    ("file", "05-process-stream.md"),
    ("file", "06-process-emotion.md"),
    ("file", "07-mu-density.md"),
    ("file", "08-su-parameter.md"),
    ("keep", "5. 被实测否决的部分"),
    ("keep", "6. 报告规范"),
    ("keep", "7. 框架明确不主张的"),
    ("file", "12-conclusion.md"),
    ("keep", "附录 A：可复现性"),
    ("keep", "附录 B：本文引用的全部数值及其来源"),
    ("file", "90-appendix-c.md"),
    ("keep", "参考文献"),
]


def read_text(path: Path) -> str:
    """Read UTF-8 and strip a BOM if present.

    A BOM on the first line makes `## 3.` fail `^## ` and silently loses the section from
    any heading-based check while leaving its body in place. That happened: a PowerShell
    `Get-Content -Raw | Set-Content` round-trip re-added a BOM to two authored sections, and
    the only symptom was one heading missing from the outline. Stripping here means the
    build cannot be broken that way again.
    """
    return path.read_text(encoding="utf-8-sig")


def split_sections(text: str) -> dict[str, str]:
    """title -> body (body excludes the `## ` line itself)."""
    out: dict[str, str] = {}
    parts = re.split(r"^(## .+)$", text, flags=re.M)
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for i in range(1, len(parts), 2):
        title = parts[i][3:].strip()
        out[title] = parts[i + 1]
    return out


def renumber(text: str, new: int) -> str:
    """Rewrite `## N.` and `### N.x` headings to the new number, and remap `§N` refs."""
    text = re.sub(r"^### (\d+)\.", lambda m: f"### {new}.", text, flags=re.M)
    text = re.sub(r"^## (\d+\.)", lambda m: f"## {new}.", text, flags=re.M)
    return text


def remap_refs(text: str, already_new: bool) -> str:
    """Remap old section references. `already_new` skips the newly authored sections,
    which were written against the NEW numbering and must not be remapped.

    CRITICAL: a reference written `PACSP-ID §8.5` points at the OTHER paper, not at this
    one. An early version of this function rewrote it to `PACSP-ID §12.5` — a citation to a
    section that does not exist in the document being cited. The negative lookbehind below
    is what prevents that, and the verifier counts those as EXTERNAL rather than unresolved.
    """
    if already_new:
        return text
    # §3.4 -> §3.3: that section's content was superseded by the rewritten §3
    text = re.sub(r"(?<!PACSP-ID )§3\.4", "§3.3", text)

    def repl(m: re.Match) -> str:
        pre, num = m.group(1), m.group(2)
        if pre:                      # "PACSP-ID §8" — not ours, leave alone
            return m.group(0)
        return "§" + REF_REMAP.get("§" + num, num)

    return re.sub(r"(PACSP-ID )?§(\d+)", repl, text)


def main() -> int:
    if not OLD.exists():
        print(f"missing {OLD}", file=sys.stderr)
        return 2
    old_text = read_text(OLD)
    old_secs = split_sections(old_text)

    pieces: list[str] = []
    kept_bytes = authored_bytes = 0
    used: list[str] = []

    for kind, name in ASSEMBLY:
        if kind == "file":
            p = SRC / name
            if not p.exists():
                print(f"missing authored section {p}", file=sys.stderr)
                return 2
            body = read_text(p)
            authored_bytes += len(body.encode("utf-8"))
            pieces.append(body.rstrip() + "\n")
            used.append(f"[new]  {name}")
            continue
        if name not in old_secs:
            print(f"section not found in 1.1.0: {name!r}", file=sys.stderr)
            print("  available:", *old_secs.keys(), sep="\n    ", file=sys.stderr)
            return 2
        body = old_secs[name]
        new_no = RENUMBER.get(name)
        if new_no is not None:
            body = renumber(body, new_no)
        body = remap_refs(body, already_new=False)
        for anchor, replacement in PATCHES.get(name, []):
            n = body.count(anchor)
            if n != 1:
                print(f"patch anchor occurs {n} times (need exactly 1) in {name!r}:",
                      file=sys.stderr)
                print(f"  anchor: {anchor[:90]!r}", file=sys.stderr)
                return 2
            body = body.replace(anchor, replacement)
        kept_bytes += len(body.encode("utf-8"))
        pieces.append(f"## {name}\n" + body.rstrip() + "\n")
        used.append(f"[keep] {name}" + (f"  -> §{new_no}" if new_no else ""))
        if new_no is not None:
            pieces[-1] = pieces[-1].replace(f"## {name}\n", f"## {new_no}. {name.split('. ', 1)[1]}\n", 1)

    out = "\n---\n\n".join(p.rstrip() + "\n" for p in pieces)
    NEW.write_text(out, encoding="utf-8")

    # ---------------- verification ----------------
    print("装配结果")
    for u in used:
        print(f"  {u}")
    print()
    print(f"  保留自 1.1.0 : {kept_bytes/1024:8.1f} KB")
    print(f"  新写        : {authored_bytes/1024:8.1f} KB")
    print(f"  合计        : {len(out.encode('utf-8'))/1024:8.1f} KB  -> {NEW.name}")

    # every §N reference must resolve to a heading that exists.
    # References written `PACSP-ID §N` point at the other paper and are counted separately.
    print("\n交叉引用自检")
    internal = set(re.findall(r"(?<!PACSP-ID )§(\d+(?:\.\d+)*)", out))
    external = set(re.findall(r"PACSP-ID §(\d+(?:\.\d+)*)", out))
    headings = set(re.findall(r"^#{2,4} (\d+(?:\.\d+)*)", out, flags=re.M))
    chapters = {str(i) for i in range(1, 13)}
    bad = []
    for r in sorted(internal):
        if r in headings or any(h.startswith(r + ".") for h in headings) or r in chapters:
            continue
        bad.append(r)
    print(f"  本文引用 {len(internal)} 个不同节号；无法解析 {len(bad)} 个")
    if bad:
        for b in bad:
            print(f"    ❌ §{b}")
    else:
        print("  ✅ 全部解析到一个存在的标题")
    if external:
        print(f"  （另有 {len(external)} 个 PACSP-ID §N 引用，指向另一篇论文，不参与本检查："
              f"{', '.join('§' + e for e in sorted(external))}）")

    # every numbered section 1..12 must actually be present as a `## N.` heading.
    # This is the check that catches a section going missing: a BOM on a heading line made
    # `^## ` fail, so the section vanished from the outline while its body stayed in place —
    # and the cross-reference check above did NOT notice, because the references still
    # resolved against the section's own sub-headings.
    print("\n节的完整性自检")
    present = {int(m) for m in re.findall(r"^## (\d+)\.", out, flags=re.M)}
    missing = sorted(set(range(1, 13)) - present)
    print(f"  应有 §1–§12；实到 {len(present)} 个")
    if missing:
        for m in missing:
            print(f"    ❌ 缺 §{m}")
    else:
        print("  ✅ §1–§12 全部存在")
    bad += [f"(missing section {m})" for m in missing]

    # report where each old section went
    print("\n旧节去向")
    for t, n in RENUMBER.items():
        if n is None:
            print(f"  {t:<44} 编号不变")
        else:
            print(f"  {t:<44} -> §{n}")
    print(f"  {'3. 表示：语料是集合，不是序列':<44} -> §3（重写）")
    print(f"  {'8. 结论':<44} -> §12（重写）")
    print(f"  {'附录 C':<44} -> 附录 C（重写）")

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
