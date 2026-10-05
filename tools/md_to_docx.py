"""Render a PACSP-M paper markdown file to DOCX, preserving the document's structure.

Usage:  python tools/md_to_docx.py [source.md] [out.docx]
        (defaults: PACSP-M-1.2.0.md -> dist/<name>.docx)

The source is hard-wrapped, which matters more than it looks. Inline emphasis frequently
spans a line break (**text on one line, continued and closed on the next**), and a parser
that works line by line cannot pair those markers, so it prints the asterisks. Twenty-eight
such spans exist. The fix is to join continuation lines into logical paragraphs before any
inline parsing happens, which is what this does: the file is first folded into blocks, and
only then are headings, tables, code, quotes, lists and formulas interpreted.

Formulas are placed in their own centred paragraph in a serif face, kept verbatim. They are
not rendered as equations; that would need an OMML generator, and a wrong equation is worse
than a readable one. The limitation is stated in the delivery.

Chinese text needs w:eastAsia set on the run in addition to the Latin font, or Word
substitutes a face with no CJK coverage. That is applied to every run.
"""

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

M = Path(__file__).resolve().parent.parent
# Default to the current paper; allow overrides so an older version can still be
# re-rendered without editing this file.
#   python tools/md_to_docx.py [source.md] [out.docx]
_src_arg = sys.argv[1] if len(sys.argv) > 1 else "PACSP-M-1.2.0.md"
_out_arg = sys.argv[2] if len(sys.argv) > 2 else None
SRC = Path(_src_arg) if Path(_src_arg).is_absolute() else M / _src_arg
OUT = (Path(_out_arg) if _out_arg and Path(_out_arg).is_absolute()
       else M / _out_arg) if _out_arg else M / "dist" / (SRC.stem + ".docx")

LATIN = "Calibri"
CJK = "微软雅黑"
MONO = "Consolas"
SERIF = "Cambria"

INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*\n]+?\*|`[^`]+`)")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_START = re.compile(r"^(#{1,6}\s|\||```|\s*[-*]\s|\s*\d+\.\s|\\\[|\\\]|\$\$)")

_JOIN_CJK = re.compile(r"([\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])$")
_JOIN_NEXT = re.compile(r"^([\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])")


def clean(t):
    return _CTRL.sub("", t)


def fold(lines):
    """Join continuation lines into logical blocks.

    Blank lines, headings, table rows, code fences and formulas each start a new block.
    Blockquote lines do not: a '>' marks a quotation, and a quotation that wraps over
    several lines is one block whose marker is stripped before joining. Treating each '>'
    line as its own block was the reason multi-line bold inside quotations stayed broken.
    List items also start a new block, since each is its own paragraph.
    """
    blocks = []
    buf = []
    in_quote = False
    in_code = False
    code = []

    def flush():
        if buf:
            blocks.append(" ".join(buf))
            buf.clear()

    for raw in lines:
        s = raw.strip()
        if s.startswith("```"):
            if in_code:
                # emit the fenced block as one marker plus its lines kept verbatim
                blocks.append("```")
                blocks.extend(code)
                blocks.append("```")
                code = []
                in_code = False
            else:
                flush()
                in_quote = False
                in_code = True
            continue
        if in_code:
            # code lines are never joined: the line breaks are the content
            code.append(raw.rstrip())
            continue
        if not s:
            flush()
            in_quote = False
            continue
        if s.startswith(">"):
            body = s.lstrip(">").strip()
            if not body:
                continue
            if not in_quote:
                flush()
                in_quote = True
            if not buf:
                buf.append(body)
            else:
                left = buf[-1]
                sep = "" if (_JOIN_CJK.search(left) and _JOIN_NEXT.search(body)) else " "
                buf[-1] = left + sep + body
            continue
        in_quote = False
        if _START.match(s):
            flush()
            blocks.append(s)
            continue
        if not buf:
            buf = [s]
            continue
        left = buf[-1]
        sep = "" if (_JOIN_CJK.search(left) and _JOIN_NEXT.search(s)) else " "
        buf[-1] = left + sep + s
    if in_code and code:
        blocks.append("```")
        blocks.extend(code)
        blocks.append("```")
    flush()
    return blocks


def set_fonts(run, latin=LATIN, cjk=CJK, size=None):
    run.font.name = latin
    if size:
        run.font.size = Pt(size)
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = rpr.makeelement(qn("w:rFonts"), {})
        rpr.insert(0, rf)
    rf.set(qn("w:ascii"), latin)
    rf.set(qn("w:hAnsi"), latin)
    rf.set(qn("w:eastAsia"), cjk)


def add_runs(par, text, base_size=10.5, latin=LATIN, cjk=CJK):
    """Split on bold, italic and code spans. Bold is tested before italic so that a
    double marker is not consumed as two single ones."""
    for part in INLINE.split(clean(text)):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = par.add_run(part[2:-2])
            r.bold = True
            set_fonts(r, latin, cjk, base_size)
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = par.add_run(part[1:-1])
            set_fonts(r, MONO, MONO, base_size - 0.5)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            r = par.add_run(part[1:-1])
            r.italic = True
            set_fonts(r, latin, cjk, base_size)
        else:
            r = par.add_run(part)
            set_fonts(r, latin, cjk, base_size)


def add_code(doc, lines):
    for i, ln in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(4 if i == 0 else 0)
        p.paragraph_format.left_indent = Cm(0.6)
        r = p.add_run(ln if ln.strip() else " ")
        set_fonts(r, MONO, MONO, 9)
        r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def latex_to_text(s):
    """Turn simple LaTeX into readable inline notation.

    Full typesetting would need an OMML generator, which is out of scope, but leaving the
    source markup on the page is worse than transliterating it: a reader should not have to
    parse \\binom{n}{2}. Only the constructs this document actually uses are handled, and
    anything unrecognised is left alone rather than guessed at.
    """
    s = s.strip()
    # Long command names must be replaced before \\left and \\right, which are prefixes of
    # \\leftarrow and \\rightsquigarrow; stripping those first left "squigarrow" behind.
    s = s.replace("\\rightsquigarrow", "↝").replace("\\leftarrow", "←")
    s = s.replace("\\rightarrow", "→").replace("\\Rightarrow", "⇒")
    s = s.replace("\\leftrightarrow", "↔")
    s = re.sub(r"\\left(?![a-zA-Z])|\\right(?![a-zA-Z])", "", s)
    s = s.replace("\\;", " ").replace("\\,", " ").replace("\\!", "")
    s = s.replace("\\quad", "    ").replace("\\qquad", "      ")
    # fractions and binomials, applied innermost-first by repeating
    for _ in range(6):
        new = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1) / (\2)", s)
        new = re.sub(r"\\binom\{([^{}]*)\}\{([^{}]*)\}", r"C(\1, \2)", new)
        new = re.sub(r"\\sqrt\{([^{}]*)\}", r"√(\1)", new)
        if new == s:
            break
        s = new
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\(?:big|Big|bigg|Bigg)(l|r)?", "", s)
    s = re.sub(r"\\exp\b", "exp", s)
    s = re.sub(r"\\log\b", "log", s)
    s = re.sub(r"\\ln\b", "ln", s)
    s = re.sub(r"\\max\b", "max", s)
    s = re.sub(r"\\min\b", "min", s)
    s = re.sub(r"\\ldots|\\dots", "…", s)
    s = re.sub(r"\\rightsquigarrow", "↝", s)
    for a, b in (
        ("\\sum", "Σ"), ("\\int", "∫"), ("\\prod", "Π"),
        ("\\le", "≤"), ("\\leq", "≤"), ("\\ge", "≥"), ("\\geq", "≥"),
        ("\\neq", "≠"), ("\\approx", "≈"), ("\\equiv", "≡"),
        ("\\times", "×"), ("\\cdot", "·"), ("\\pm", "±"),
        ("\\to", "→"), ("\\rightarrow", "→"), ("\\Rightarrow", "⇒"),
        ("\\in", "∈"), ("\\subset", "⊂"), ("\\cup", "∪"), ("\\cap", "∩"),
        ("\\infty", "∞"), ("\\partial", "∂"), ("\\nabla", "∇"),
        ("\\alpha", "α"), ("\\beta", "β"), ("\\gamma", "γ"), ("\\delta", "δ"),
        ("\\epsilon", "ε"), ("\\mu", "μ"), ("\\nu", "ν"), ("\\sigma", "σ"),
        ("\\tau", "τ"), ("\\phi", "φ"), ("\\varphi", "φ"), ("\\Phi", "Φ"),
        ("\\Psi", "Ψ"), ("\\varepsilon", "ε"),
        ("\\omega", "ω"), ("\\Omega", "Ω"), ("\\Lambda", "Λ"), ("\\lambda", "λ"),
        ("\\Pi", "Π"), ("\\pi", "π"), ("\\theta", "θ"), ("\\kappa", "κ"),
        ("\\rho", "ρ"), ("\\eta", "η"), ("\\zeta", "ζ"), ("\\chi", "χ"),
        ("\\|", "‖"), ("\\{", "{"), ("\\}", "}"),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s)
    # thin-space commands leave a backslash behind when they follow a comma, and in plain
    # text that reads as an escape. Space after punctuation is already handled, so drop it.
    s = re.sub(r"([,;:])\s*\\\s*", r"\1 ", s)
    s = re.sub(r"\\\s+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def add_math(doc, lines):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(latex_to_text(" ".join(lines)))
    set_fonts(r, SERIF, SERIF, 11)
    r.italic = True


def parse_table_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def add_table(doc, rows):
    if not rows:
        return
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            par = t.cell(i, j).paragraphs[0]
            par.paragraph_format.space_before = Pt(1)
            par.paragraph_format.space_after = Pt(1)
            add_runs(par, cell, base_size=9)
            if i == 0:
                for r in par.runs:
                    r.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def main():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    blocks = fold(lines)
    doc = Document()

    st = doc.styles["Normal"]
    st.font.name = LATIN
    st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK)

    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.0)
        s.top_margin = s.bottom_margin = Cm(1.8)

    i, n = 0, len(blocks)
    while i < n:
        b = blocks[i]

        if b.startswith("```"):
            code = []
            i += 1
            while i < n and not blocks[i].startswith("```"):
                code.append(blocks[i])
                i += 1
            add_code(doc, code)
            i += 1
            continue

        if b in ("$$", "\\[", "\\]"):
            math = []
            i += 1
            while i < n and blocks[i] not in ("$$", "\\]", "\\["):
                math.append(blocks[i])
                i += 1
            add_math(doc, math)
            i += 1
            continue

        if b == "---":
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", b)
        if m:
            lvl, body = len(m.group(1)), m.group(2)
            if lvl == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(14)
                add_runs(p, body, base_size=19)
                for r in p.runs:
                    r.bold = True
            else:
                p = doc.add_heading(level=min(lvl - 1, 4))
                for r in list(p.runs):
                    r.text = ""
                add_runs(p, body, base_size={2: 15, 3: 12.5, 4: 11.5}.get(lvl, 11))
                for r in p.runs:
                    r.bold = True
                    r.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
            i += 1
            continue

        if b.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|$", blocks[i + 1]):
            rows = [parse_table_row(b)]
            i += 2
            while i < n and blocks[i].startswith("|"):
                rows.append(parse_table_row(blocks[i]))
                i += 1
            add_table(doc, rows)
            continue

        if b.startswith(">"):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.right_indent = Cm(0.5)
            p.paragraph_format.space_before = Pt(5)
            p.paragraph_format.space_after = Pt(5)
            add_runs(p, b.lstrip("> ").strip(), base_size=10.5)
            i += 1
            continue

        mb = re.match(r"^([-*])\s+(.*)$", b)
        if mb:
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.left_indent = Cm(0.7)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, mb.group(2))
            i += 1
            continue

        mn = re.match(r"^(\d+)\.\s+(.*)$", b)
        if mn:
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.left_indent = Cm(0.7)
            p.paragraph_format.space_after = Pt(2)
            add_runs(p, mn.group(2))
            i += 1
            continue

        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        add_runs(p, b)
        i += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"  source lines   : {len(lines)}")
    print(f"  logical blocks : {len(blocks)}")
    print(f"  wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    print(f"  paragraphs {len(doc.paragraphs)}  tables {len(doc.tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
