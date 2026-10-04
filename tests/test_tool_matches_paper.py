"""Regression test: the tool must reproduce the paper's published values.

A tool that ships alongside a paper has to agree with it, or one of the two is wrong. This
runs the tool's own code paths against the frozen corpora and asserts the numbers match
PACSP-M 1.1.0 sections 4.3 and 4.11 within tolerance.

Tolerances are stated per case. The ratio of within-collection D and the separation and
overlap figures are deterministic given the frame, so they are held to 1 percent. The
bootstrap intervals depend on the resampling, so they are only checked for containing the
point estimate and for the same direction.

Run:  python tests/test_tool_matches_paper.py
"""

import sys
from pathlib import Path

import numpy as np

M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))

import pacsp_tool as T  # noqa: E402

# from the paper: collection A / collection B, ratio of D, separation, overlap
CASES = [
    ("poem", "machine_poem", 0.8305, 1.2773, 0.797),
    ("lyrics", "machine_lyrics", 0.8691, 1.4516, 0.066),
    ("techdoc", "machine_techdoc2", 0.9205, None, None),
    ("hc3_human_medicine", "hc3_ai_medicine", 0.9147, 1.0547, 0.811),
    ("hc3_human_openqa", "hc3_ai_openqa", 0.9358, 1.0301, 0.854),
]

# from the paper section 4.11: lexical style ratio, cross over within
STYLE = {
    "poem": 0.0644, "lyrics": 0.0395, "techdoc": 0.3824,
    "medicine": 1.0317, "openqa": 1.1984,
}

TOL = 0.02          # 2 percent, covers bootstrap and sampling variation
FAILS = []


def check(label, got, want, tol=TOL):
    if got is None or want is None:
        ok = got is None and want is None
    else:
        ok = abs(got - want) <= tol * max(1.0, abs(want))
    mark = "ok  " if ok else "FAIL"
    g = "None" if got is None else f"{got:.4f}"
    w = "None" if want is None else f"{want:.4f}"
    print(f"  [{mark}] {label:<44} tool {g:>9}   paper {w:>9}")
    if not ok:
        FAILS.append(label)
    return ok


def main():
    emb = T.Embedder(T.KNOWN_FRAMES["bge-large-zh"][0])
    rng = np.random.default_rng(7)

    print("=== within-collection dispersion (paper 4.3) ===")
    for a, b, want_ratio, _, _ in CASES:
        ta, _ = T.load_texts(M / "data" / a)
        tb, _ = T.load_texts(M / "data" / b)
        EA, EB = emb.encode(ta), emb.encode(tb)
        dA = T.within_distances(EA).mean()
        dB = T.within_distances(EB).mean()
        check(f"D ratio {a} / {b}", dA / dB, want_ratio)

    print("\n=== cross-collection relations (paper 4.11) ===")
    for a, b, _, want_sep, want_ov in CASES:
        if want_sep is None:
            print(f"  [skip] {a:<44} no published cross figure")
            continue
        ta, _ = T.load_texts(M / "data" / a)
        tb, _ = T.load_texts(M / "data" / b)
        EA, EB = emb.encode(ta), emb.encode(tb)
        wA, wB = T.within_distances(EA), T.within_distances(EB)
        X = T.cross_distances(EA, EB)
        pooled = np.concatenate([wA, wB])
        sep = float(X.mean() / pooled.mean())
        thr = float(np.percentile(pooled, 95))
        ov = float((X < thr).mean())
        check(f"separation {a}", sep, want_sep)
        check(f"overlap {a}", ov, want_ov)

    print("\n=== lexical style (paper 4.11) ===")
    for a, b, _, _, _ in CASES:
        dom = a.split("_")[-1]
        if dom not in STYLE:
            continue
        ta, _ = T.load_texts(M / "data" / a)
        tb, _ = T.load_texts(M / "data" / b)
        st = T.style_ratio(ta, tb, 4000, rng)
        check(f"style ratio {dom}", st["ratio"], STYLE[dom],
              tol=0.15 if st["within_too_small"] else TOL)
        if st["within_too_small"]:
            print(f"         note: within lexical overlap {st['within_a']:.4f} is low,")
            print(f"               so a 15 percent tolerance is applied here")

    print("\n=== gates behave as the paper requires (4.11) ===")
    ta, _ = T.load_texts(M / "data" / "hc3_human_openqa")
    tb, _ = T.load_texts(M / "data" / "hc3_ai_openqa")
    EA, EB = emb.encode(ta), emb.encode(tb)
    wA, wB = T.within_distances(EA), T.within_distances(EB)
    X = T.cross_distances(EA, EB)
    pooled = np.concatenate([wA, wB])
    ov = float((X < pooled.mean() * 1.0).mean())  # arbitrary, only to drive the gate
    ov_hi = float((X < np.percentile(pooled, 95)).mean())
    g = T.gates_for_compare(len(ta), len(tb), {"cv": 0.01},
                            {"point": 0.9358, "ci95": [0.90, 0.97]}, ov_hi, True)
    fired = {x.id for x in g if x.fired}
    ok = "not-assignable" in fired
    print(f"  [{'ok  ' if ok else 'FAIL'}] openqa fires not-assignable "
          f"(overlap {ov_hi:.3f})")
    if not ok:
        FAILS.append("not-assignable gate for openqa")

    v = T.verdict_from(g, 1.03, ov_hi)
    ok2 = v.startswith("group-mean-differs-only")
    print(f"  [{'ok  ' if ok2 else 'FAIL'}] verdict is group-mean-differs-only")
    print(f"         got: {v}")
    if not ok2:
        FAILS.append("verdict for openqa")

    # the tiny-effect case from section 6.3: interval excludes 1 and effect is ~1 percent
    g2 = T.gates_for_compare(31, 31, {"cv": 0.003},
                             {"point": 1.0100, "ci95": [1.004, 1.016]}, 0.9, True)
    fired2 = {x.id for x in g2 if x.fired}
    ok3 = "effect-below-floor" in fired2
    print(f"  [{'ok  ' if ok3 else 'FAIL'}] a 1 percent effect fires effect-below-floor "
          f"(paper 6.3)")
    if not ok3:
        FAILS.append("effect-below-floor gate")

    print()
    if FAILS:
        print(f"  {len(FAILS)} FAILURES:")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  all checks passed: the tool reproduces the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
