"""pacsp — measure and compare semantic dispersion of text collections.

This packages the PACSP-M framework into a command line tool. Its design follows the
paper's two principles, and the second one is what makes it more than a calculator:

    P1  traceability   every reported number traces to a declared frame, and is
                       insensitive to undeclared choices
    P1' sensitivity    the number must be sensitive to the difference it claims to measure

P1' is enforced by gates. A tool that prints a ratio lets a user claim separability from a
difference in group means; the evidence does not support that, so this tool says so in the
output rather than leaving the reader to infer it.

Commands
    frame     describe a frame and check it is usable for comparison
    compare   relations between two collections: interval, separation, overlap, style
    measure   within-collection dispersion
    selfcheck run the gates on a single collection

Usage
    python pacsp_tool.py compare A/ B/ --name-a human --name-b machine
    python pacsp_tool.py measure A/ --json out.json
    python pacsp_tool.py frame --list

Every comparison prints the statistic, its interval, the gates that fired, and a verdict
limited to what the evidence supports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

KNOWN_FRAMES = {
    "bge-large-zh": ("BAAI/bge-large-zh-v1.5", "zh", 1024,
                     "checked: 5/5 direction, spread 1.13"),
    "bge-small-zh": ("BAAI/bge-small-zh-v1.5", "zh", 512,
                     "checked: 5/5 direction, spread 1.12; same family as bge-large"),
    "text2vec-base-zh": ("shibing624/text2vec-base-chinese", "zh", 768,
                         "NOT checked as a comparison frame: 3/5 direction, effect "
                         "shrinks to about 1 percent"),
}
DEFAULT_FRAME = "bge-large-zh"

# Gates. Each is a condition under which a claim must not be made.
GATE_SD_CEILING = 0.15        # above this, the estimate is too imprecise to compare
GATE_EFFECT_FLOOR = 0.02      # below this, a ratio is indistinguishable from 1
GATE_MIN_N = 8                # below this, a bootstrap interval is not meaningful
GATE_MIN_PAIRS_DIR = 3        # below this, no direction consistency can be claimed

TEXT_SUFFIXES = {".txt", ".md", ".text", ".csv", ".tsv"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_texts(path: str | Path) -> tuple[list[str], list[str]]:
    """Read a collection from a directory of text files or from a single file.

    A single file is split on blank lines, which is the common shape for a corpus kept as
    one document. Directory files are read whole. Empty items are dropped, since an empty
    string embeds to a meaningless vector and would silently enter every distance.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.is_file()
                       and f.suffix.lower() in TEXT_SUFFIXES)
        out, names = [], []
        for f in files:
            t = f.read_text(encoding="utf-8", errors="replace").strip()
            if t:
                out.append(t)
                names.append(f.name)
        return out, names
    if p.is_file():
        raw = p.read_text(encoding="utf-8", errors="replace")
        parts = [x.strip() for x in re.split(r"\n\s*\n", raw) if x.strip()]
        return parts, [f"{p.name}#{i}" for i in range(len(parts))]
    raise FileNotFoundError(f"no such file or directory: {path}")


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedder:
    """Caches the model and the vectors it has already produced for this process.

    Two backends are available. sentence-transformers is the original path and needs torch,
    about 5.4 GB; the ONNX backend needs onnxruntime instead, roughly 330 MB with the model.
    They agree to four decimals on the published ratios, which is what
    tools/onnx_gpu_acceptance.py checks, so the default is onnx when a graph is present and
    sentence-transformers otherwise.
    """

    def __init__(self, model_id: str, backend: str = "auto",
                 onnx_dir: str | Path | None = None, prefer_gpu: bool = True):
        self.model_id = model_id
        self.onnx_dir = Path(onnx_dir) if onnx_dir else Path(__file__).resolve().parent / "onnx"
        self.backend = backend
        self.prefer_gpu = prefer_gpu
        self._impl = None
        self._cache: dict = {}

    def _load(self):
        if self._impl is not None:
            return self._impl
        import pacsp_backends
        chosen = self.backend
        if chosen == "auto":
            graph = self.onnx_dir / "bge-large-zh-v1.5.onnx"
            chosen = "onnx" if graph.exists() else "sentence-transformers"
        try:
            self._impl = pacsp_backends.make_embedder(
                chosen, self.model_id, onnx_dir=self.onnx_dir,
                prefer_gpu=self.prefer_gpu)
        except Exception as e:
            if chosen == "onnx" and self.backend == "auto":
                print(f"  note: ONNX backend unavailable ({type(e).__name__}), "
                      "falling back to sentence-transformers", file=sys.stderr)
                self._impl = pacsp_backends.make_embedder(
                    "sentence-transformers", self.model_id)
            else:
                raise
        return self._impl

    @property
    def active_backend(self) -> str:
        return self._load().name

    def describe(self) -> dict:
        return self._load().describe()

    def encode(self, texts: list[str]) -> np.ndarray:
        impl = self._load()
        key = (impl.name, len(texts), len(texts[0]) if texts else 0,
               hash(tuple(len(t) for t in texts)))
        if key in self._cache:
            return self._cache[key]
        E = impl.encode(texts)
        self._cache[key] = E
        return E


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0))


def within_distances(E: np.ndarray) -> np.ndarray:
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return sqdist(E, E)[iu]


def cross_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return sqdist(A, B).ravel()


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, rng: np.random.Generator,
                      alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Return point, lo, hi, and the coefficient of variation of the bootstrap means.

    The CV reported is of the resampled means, which measures how precisely the mean is
    pinned down by this sample. It is the quantity gate one uses. It is not the CV of the
    underlying distances, which measures how spread the data are.
    """
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    cv = float(means.std() / means.mean()) if means.mean() else float("inf")
    return float(x.mean()), float(lo), float(hi), cv


# ---------------------------------------------------------------------------
# Style, independent of the frame
# ---------------------------------------------------------------------------

def char_ngrams(t: str, n: int = 2) -> set:
    s = " ".join(t.split())
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def style_ratio(texts_a: list[str], texts_b: list[str], sample: int,
                rng: np.random.Generator) -> dict:
    """Lexical overlap between the collections against overlap inside collection A.

    Two sampling details decide whether this means anything.

    Self-pairs are excluded. An item compared with itself scores Jaccard 1.0, so including
    the diagonal inflates the within figure toward 1 and drags the ratio toward 1 with it;
    that alone turned a ratio of 0.04 into one of 1.01 during testing.

    Duplicate pairs are excluded too, for the same reason: drawing the same unordered pair
    twice over-weights it. Pairs are drawn as a set of index pairs.

    The full cross product is quadratic, so pairs are sampled rather than enumerated. A
    value near 1 means the two collections share vocabulary to the degree one collection
    shares it with itself; a small value means they barely do.
    """
    ga = [char_ngrams(t) for t in texts_a]
    gb = [char_ngrams(t) for t in texts_b]
    na, nb = len(ga), len(gb)

    def sample_pairs(n_i, n_j, same):
        seen, out = set(), []
        cap = 40 * sample
        tries = 0
        while len(out) < sample and tries < cap:
            tries += 1
            i = int(rng.integers(0, n_i))
            j = int(rng.integers(0, n_j))
            if same and i == j:
                continue
            key = (i, j) if i <= j else (j, i)
            if key in seen:
                continue
            seen.add(key)
            out.append((i, j))
        return out

    wp = sample_pairs(na, na, True)
    cp = sample_pairs(na, nb, False)
    w = np.asarray([jaccard(ga[i], ga[j]) for i, j in wp]) if wp else np.array([0.0])
    c = np.asarray([jaccard(ga[i], gb[j]) for i, j in cp]) if cp else np.array([0.0])

    ratio = float(c.mean() / w.mean()) if w.mean() > 0 else None
    return {"within_a": float(w.mean()), "cross": float(c.mean()),
            "ratio": ratio,
            "pairs_within": len(wp), "pairs_cross": len(cp),
            # below this the within figure is too near zero for the ratio to be stable
            "within_too_small": bool(w.mean() < 0.02)}


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    id: str
    fired: bool
    detail: str


def gates_for_compare(n_a: int, n_b: int, sep: dict, ratio_ci: dict,
                      overlap: float, direction_ok: bool,
                      style: dict | None = None) -> list[Gate]:
    g = []

    if min(n_a, n_b) < GATE_MIN_N:
        g.append(Gate("sample-too-small", True,
                      f"n = {n_a} vs {n_b}; intervals need at least {GATE_MIN_N} per side"))
    else:
        g.append(Gate("sample-too-small", False, f"n = {n_a} vs {n_b}"))

    cv = sep.get("cv", 0.0)
    if cv > GATE_SD_CEILING:
        g.append(Gate("estimate-imprecise", True,
                      f"bootstrap CV of the cross mean is {cv:.4f} > {GATE_SD_CEILING}; "
                      "do not compare collections"))
    else:
        g.append(Gate("estimate-imprecise", False, f"bootstrap CV {cv:.4f}"))

    # a ratio within the effect floor of 1 cannot be distinguished from no difference
    r = ratio_ci.get("point")
    lo, hi = ratio_ci.get("ci95", [None, None])
    if r is None:
        g.append(Gate("effect-unmeasurable", True, "ratio undefined"))
    elif lo is not None and hi is not None and not (lo > 1 or hi < 1):
        g.append(Gate("interval-includes-1", True,
                      f"ratio {r:.4f} CI [{lo:.4f}, {hi:.4f}] includes 1: "
                      "no direction can be claimed"))
    else:
        small = abs(r - 1) < GATE_EFFECT_FLOOR
        g.append(Gate("interval-includes-1", False,
                      f"ratio {r:.4f} CI [{lo:.4f}, {hi:.4f}]"))
        if small:
            g.append(Gate("effect-below-floor", True,
                          f"|ratio - 1| = {abs(r - 1):.4f} < {GATE_EFFECT_FLOOR}: the "
                          "interval may exclude 1 while the effect is too small to act on"))

    # the assignability gate: this is the one that stops over-reading a mean difference
    if overlap > 0.5:
        g.append(Gate("not-assignable", True,
                      f"overlap share {overlap:.3f} > 0.5: the collections interpenetrate, "
                      "so individual items cannot be assigned to a side from this statistic"))
    else:
        g.append(Gate("not-assignable", False,
                      f"overlap share {overlap:.3f}: some assignability possible"))

    if not direction_ok:
        g.append(Gate("direction-inconsistent", True,
                      "within-collection ratios do not agree in direction"))
    else:
        g.append(Gate("direction-inconsistent", False, "within-collection ratios agree"))

    if style is not None:
        if style.get("ratio") is None:
            g.append(Gate("style-unmeasurable", True,
                          "within-collection lexical overlap is zero; the ratio is undefined"))
        elif style.get("within_too_small"):
            g.append(Gate("style-unstable", True,
                          f"within-collection lexical overlap is only "
                          f"{style['within_a']:.4f}; the ratio {style['ratio']:.4f} is "
                          "numerically unstable and should not be reported as a value"))
        else:
            g.append(Gate("style-unstable", False,
                          f"within lexical overlap {style['within_a']:.4f}, "
                          f"ratio {style['ratio']:.4f}"))

    return g


def verdict_from(gates: list[Gate], sep: float, overlap: float) -> str:
    fired = {g.id for g in gates if g.fired}
    if "sample-too-small" in fired or "estimate-imprecise" in fired:
        return "insufficient-evidence: fix sampling before drawing any conclusion"
    if "interval-includes-1" in fired:
        return "no-detectable-difference: the collections are not distinguishable here"
    if "not-assignable" in fired:
        return ("group-mean-differs-only: the mean dispersion differs, but the "
                "collections interpenetrate so single items cannot be classified")
    return "separable: mean differs and collections are largely disjoint"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def resolve_frame(args) -> tuple[str, str, int, str]:
    key = args.frame
    if key in KNOWN_FRAMES:
        repo, lang, dim, note = KNOWN_FRAMES[key]
        return repo, lang, dim, note
    # an arbitrary sentence-transformers id is allowed, but must be declared
    return key, "unknown", 0, "unchecked frame: no direction validation for this model"


def cmd_frame(args) -> int:
    if args.list:
        print("known frames:\n")
        for k, (repo, lang, dim, note) in KNOWN_FRAMES.items():
            mark = " *" if k == DEFAULT_FRAME else "  "
            print(f"{mark} {k:<20} dim {dim:<5} lang {lang:<4} {repo}")
            print(f"      {note}")
        print("\n  * default. Any sentence-transformers id may be passed with --frame,")
        print("    but an unchecked frame invalidates cross-frame comparison.")
        return 0

    repo, lang, dim, note = resolve_frame(args)
    print(f"frame      : {args.frame}")
    print(f"repository : {repo}")
    print(f"language   : {lang}")
    print(f"note       : {note}")
    if args.frame not in KNOWN_FRAMES:
        print("\nWARNING: this frame has no direction validation in PACSP-M.")
        print("Report it as declared but unvalidated; do not compare it with a")
        print("validated frame. See the paper, section 4.9.")
    return 0


def _measure_core(E: np.ndarray, n_boot: int, rng: np.random.Generator) -> dict:
    w = within_distances(E)
    point, lo, hi, cv = bootstrap_mean_ci(w, n_boot, rng)
    return {"n": int(len(E)), "D": round(point, 6),
            "ci95": [round(lo, 6), round(hi, 6)], "bootstrap_cv": round(cv, 6),
            "n_distances": int(len(w))}


def cmd_measure(args) -> int:
    repo, lang, dim, note = resolve_frame(args)
    emb = Embedder(repo, backend=args.backend, onnx_dir=args.onnx_dir,
                    prefer_gpu=not args.no_gpu)
    rng = np.random.default_rng(args.seed)

    texts, names = load_texts(args.collection)
    if len(texts) < GATE_MIN_N:
        print(f"  only {len(texts)} texts; need at least {GATE_MIN_N}", file=sys.stderr)
        return 2
    E = emb.encode(texts)
    res = _measure_core(E, args.bootstrap, rng)
    res["frame"] = {"name": args.frame, "repository": repo, "language": lang,
                    "note": note}

    gates = []
    if res["bootstrap_cv"] > GATE_SD_CEILING:
        gates.append(Gate("estimate-imprecise", True,
                          f"bootstrap CV {res['bootstrap_cv']:.4f} > {GATE_SD_CEILING}"))
    else:
        gates.append(Gate("estimate-imprecise", False,
                          f"bootstrap CV {res['bootstrap_cv']:.4f}"))

    if args.json:
        Path(args.json).write_text(
            json.dumps({**res, "gates": [asdict(g) for g in gates]},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written {args.json}")

    print(f"\n  collection : {args.collection}  ({len(texts)} texts)")
    print(f"  frame      : {args.frame}  ({repo})")
    print(f"  D          : {res['D']:.4f}  95% CI [{res['ci95'][0]:.4f}, "
          f"{res['ci95'][1]:.4f}]")
    print(f"  precision  : bootstrap CV {res['bootstrap_cv']:.4f} "
          f"(gate: <= {GATE_SD_CEILING})")
    print(f"  distances  : {res['n_distances']}")
    for g in gates:
        print(f"  [{'X' if g.fired else 'ok'}] {g.id}: {g.detail}")
    print("\n  D has no meaning outside this frame. Ratios of D across frames are")
    print("  not comparable; see the paper, section 4.9.")
    return 0


def cmd_compare(args) -> int:
    repo, lang, dim, note = resolve_frame(args)
    emb = Embedder(repo, backend=args.backend, onnx_dir=args.onnx_dir,
                    prefer_gpu=not args.no_gpu)
    rng = np.random.default_rng(args.seed)

    ta, na = load_texts(args.collection_a)
    tb, nb = load_texts(args.collection_b)
    for label, t in ((args.name_a, ta), (args.name_b, tb)):
        if len(t) < GATE_MIN_N:
            print(f"  {label}: only {len(t)} texts; need at least {GATE_MIN_N}",
                  file=sys.stderr)
            return 2

    EA, EB = emb.encode(ta), emb.encode(tb)
    wA, wB = within_distances(EA), within_distances(EB)
    X = cross_distances(EA, EB)

    m_a = _measure_core(EA, args.bootstrap, rng)
    m_b = _measure_core(EB, args.bootstrap, rng)

    # ratio of within-collection D, with an interval from resampling items
    ratio_boot = []
    for _ in range(args.bootstrap):
        ia = rng.integers(0, len(EA), len(EA))
        ib = rng.integers(0, len(EB), len(EB))
        da = within_distances(EA[ia]).mean()
        db = within_distances(EB[ib]).mean()
        ratio_boot.append(da / db)
    ratio_boot = np.asarray(ratio_boot)
    ratio = {"point": float(m_a["D"] / m_b["D"]),
             "ci95": [float(np.percentile(ratio_boot, 2.5)),
                      float(np.percentile(ratio_boot, 97.5))]}

    cross_mean, cross_lo, cross_hi, cross_cv = bootstrap_mean_ci(
        X, args.bootstrap, rng)
    pooled = np.concatenate([wA, wB])
    separation = float(X.mean() / pooled.mean())
    thr = float(np.percentile(pooled, 95))
    overlap = float((X < thr).mean())

    style = style_ratio(ta, tb, args.style_pairs, rng) if args.style else None

    direction_ok = (ratio["ci95"][1] < 1) or (ratio["ci95"][0] > 1)
    gates = gates_for_compare(len(ta), len(tb),
                              {"cv": cross_cv}, ratio, overlap, direction_ok,
                              style=style)
    verdict = verdict_from(gates, separation, overlap)

    out = {
        "frame": {"name": args.frame, "repository": repo, "language": lang,
                  "note": note},
        "collections": {
            "a": {"path": str(args.collection_a), "name": args.name_a,
                  "n": len(ta), "D": m_a["D"], "ci95": m_a["ci95"],
                  "bootstrap_cv": m_a["bootstrap_cv"]},
            "b": {"path": str(args.collection_b), "name": args.name_b,
                  "n": len(tb), "D": m_b["D"], "ci95": m_b["ci95"],
                  "bootstrap_cv": m_b["bootstrap_cv"]},
        },
        "interval": {"cross_mean": round(cross_mean, 6),
                     "ci95": [round(cross_lo, 6), round(cross_hi, 6)],
                     "bootstrap_cv": round(cross_cv, 6),
                     "n_cross_distances": int(len(X))},
        "separation": {"value": round(separation, 6),
                       "definition": "cross mean / pooled within mean"},
        "overlap": {"value": round(overlap, 6),
                    "radius": round(thr, 6),
                    "definition": "share of cross distances below the pooled "
                                  "within-distance 95th percentile"},
        "ratio_within": {"value": round(ratio["point"], 6),
                         "ci95": [round(ratio["ci95"][0], 6),
                                  round(ratio["ci95"][1], 6)]},
        "style": style,
        "gates": [asdict(g) for g in gates],
        "verdict": verdict,
        "limits": [
            "D and all distances are frame-dependent; compare only within one frame",
            "the overlap radius is the pooled within-distance P95; other radii shift it",
            "cross distances are not independent, so intervals are approximate",
            "style is lexical char-2gram Jaccard, a coarse and independent axis",
        ],
    }

    if args.json:
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
        print(f"  written {args.json}")

    print(f"\n  frame            : {args.frame}  ({repo})")
    print(f"  {args.name_a:<16}: n={len(ta):<4} D={m_a['D']:.4f} "
          f"CI[{m_a['ci95'][0]:.4f}, {m_a['ci95'][1]:.4f}]")
    print(f"  {args.name_b:<16}: n={len(tb):<4} D={m_b['D']:.4f} "
          f"CI[{m_b['ci95'][0]:.4f}, {m_b['ci95'][1]:.4f}]")
    print(f"\n  interval         : cross mean {cross_mean:.4f} "
          f"CI[{cross_lo:.4f}, {cross_hi:.4f}]  CV {cross_cv:.4f}")
    print(f"  separation       : {separation:.4f}   (cross / pooled within)")
    print(f"  overlap          : {overlap:.4f}   (radius {thr:.4f}, "
          f"{len(X)} cross distances)")
    print(f"  ratio of D       : {ratio['point']:.4f} "
          f"CI[{ratio['ci95'][0]:.4f}, {ratio['ci95'][1]:.4f}]")
    if style:
        print(f"  style (lexical)  : cross/within = {style['ratio']:.4f}  "
              f"(within {style['within_a']:.4f}, cross {style['cross']:.4f})")
    print("\n  gates:")
    for g in gates:
        print(f"    [{'X' if g.fired else 'ok'}] {g.id:<24} {g.detail}")
    print(f"\n  VERDICT: {verdict}")
    if "group-mean-differs-only" in verdict:
        print("\n  The mean differs but the collections interpenetrate. Do NOT report")
        print("  that items can be classified, or that the groups occupy different")
        print("  regions. See the paper, section 4.11.")
    return 0


def cmd_selfcheck(args) -> int:
    """Run the gates on one collection and report what may be claimed about it."""
    return cmd_measure(args)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pacsp",
        description="Measure and compare semantic dispersion of text collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Commands")[1] if "Commands" in __doc__ else None)
    p.add_argument("--frame", default=DEFAULT_FRAME,
                   help=f"embedding frame (default {DEFAULT_FRAME}; "
                        "any sentence-transformers id also accepted)")
    p.add_argument("--bootstrap", type=int, default=2000,
                   help="bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    p.add_argument("--backend", default="auto",
                   choices=["auto", "onnx", "sentence-transformers"],
                   help="embedding backend; auto prefers onnx when a graph is present")
    p.add_argument("--onnx-dir", default=None,
                   help="directory holding bge-large-zh-v1.5.onnx")
    p.add_argument("--no-gpu", action="store_true",
                   help="do not look for torch's bundled CUDA libraries")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frame", help="describe a frame and its validation status")
    f.add_argument("--list", action="store_true", help="list known frames")
    f.set_defaults(func=cmd_frame)

    m = sub.add_parser("measure", help="within-collection dispersion")
    m.add_argument("collection")
    m.add_argument("--json", help="write the result as JSON")
    m.set_defaults(func=cmd_measure)

    s = sub.add_parser("selfcheck", help="run the gates on one collection")
    s.add_argument("collection")
    s.add_argument("--json", help="write the result as JSON")
    s.set_defaults(func=cmd_selfcheck)

    c = sub.add_parser("compare", help="relations between two collections")
    c.add_argument("collection_a")
    c.add_argument("collection_b")
    c.add_argument("--name-a", default="A")
    c.add_argument("--name-b", default="B")
    c.add_argument("--style", action="store_true", default=True,
                   help="also compute lexical style overlap (default on)")
    c.add_argument("--no-style", dest="style", action="store_false")
    c.add_argument("--style-pairs", type=int, default=4000,
                   help="sampled pairs for the style measure (default 4000)")
    c.add_argument("--json", help="write the result as JSON")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
