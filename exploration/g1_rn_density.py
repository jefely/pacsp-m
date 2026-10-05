"""G1: give mu a FAITHFUL per-cell definition via the Radon-Nikodym density, then measure
whether it fixes order sensitivity.

THE DEFECT THIS ADDRESSES
-------------------------
The theory says C_T = integral of mu(t) dLambda(t) with mu a path-local quantity. The
implementation computes C_T = sum_k mu_k*delta_k where mu_k = 1 - mean cosine similarity
*inside a window*. That is a category error, and it is worth stating precisely:

    mu_k is a dispersion measured over a NEIGHBOURHOOD of cells,
    but it is used as a WEIGHT on a SINGLE cell.

A weight that multiplies one cell must be a property of that cell. This is why localising
the window made things worse rather than better (radius 1 is worse than radius 5): shrinking
the window sharpens the mismatch instead of removing it.

THE FAITHFUL FORM
-----------------
Radon-Nikodym says a density is a ratio of two measures ON THE SAME CELLS:

    rho_k = Lambda(cell_k) / nu(cell_k)

with Lambda the path's own measure and nu a reference measure (the framework's own text
calls nu the "public cognitive substrate", and writes dLambda = rho dnu + dM). Because rho_k
is defined at cell k from cell k's own two masses, it is cell-local BY CONSTRUCTION — not a
neighbourhood quantity pressed into service as a local weight.

Under this reading the framework's integral becomes

    C_T = integral rho dLambda = integral rho^2 dnu = sum_k delta_k^2 / nu_k

and the old implementation's mu = 1 is no longer "the weight dropped" but the degenerate
choice nu = Lambda, for which rho = dLambda/dLambda = 1 identically. That relabels the
question: not "keep mu or drop it", but "which reference measure is Lambda compared against".

VERIFICATION ANCHOR (an identity, not a tolerance)
--------------------------------------------------
rho must integrate back to Lambda over nu:  sum_k rho_k*nu_k == Lambda(total) == sum_k delta_k.
Any variant failing this is not a density and is rejected before its C_T is even measured.

Run:  python PACSP-M/exploration/g1_rn_density.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
sys.path.insert(0, str(M.parent / "toolkit" / "math"))
import pacsp_core  # noqa: E402

try:
    from pacsp_math.lebesgue import lebesgue_decompose  # noqa: E402
except Exception:                                              # noqa: BLE001
    lebesgue_decompose = None

CACHE = M / "onnx" / "embcache"
NPERM = 200
ARMS = ["poem", "machine_poem", "lyrics", "techdoc",
        "hc3_human_medicine", "hc3_ai_medicine"]

EXISTING = ["global", "window", "narrow", "adjacent", "none"]
RN = ["rn_uniform", "rn_iso", "rn_centroid", "rn_selfnorm"]
ALL = EXISTING + RN


# ---------------------------------------------------------------- measures

def delta_of(E: np.ndarray) -> np.ndarray:
    """The path's own measure: mass on each cell, from the shipped implementation."""
    return np.asarray(pacsp_core.compute_deltas(E), dtype=float)


def reference(E: np.ndarray, kind: str) -> tuple[np.ndarray, str]:
    """The reference measure nu on the SAME cells as delta, plus a human label.

    nu must be defined cell-by-cell. Anything computed over a neighbourhood is not admitted
    here — that is the whole point of the exercise.
    """
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    n = len(N)
    if kind == "rn_uniform":
        return np.ones(n - 1), "nu = 1  (counting measure; the isotropic reference)"
    if kind == "rn_iso":
        # mean pairwise distance: what a step would look like between unrelated items
        d = np.linalg.norm(N[:, None, :] - N[None, :, :], axis=2)
        iu = np.triu_indices(n, k=1)
        return np.full(n - 1, float(d[iu].mean())), "nu = mean pairwise distance (isotropic null)"
    if kind == "rn_centroid":
        g = N.mean(axis=0)
        dist = np.linalg.norm(N - g, axis=1)
        nu = (dist[1:] + dist[:-1]) / 2.0
        return np.maximum(nu, 1e-12), "nu = mean distance of the cell's two endpoints to the centroid"
    if kind == "rn_selfnorm":
        d = delta_of(E)
        return np.full(len(d), float(d.sum())), "nu = Lambda(total) (the path's own total mass)"
    raise ValueError(kind)


def rn_density(E: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """(rho, nu, delta, label) for a faithful per-cell density. Raises if the identity fails."""
    d = delta_of(E)
    nu, label = reference(E, kind)
    m = min(len(d), len(nu))
    d, nu = d[:m], nu[:m]
    rho = d / nu
    # THE identity: a density must integrate back to the measure it is a density of
    if not np.isclose(float(np.sum(rho * nu)), float(np.sum(d)), rtol=1e-9, atol=1e-9):
        raise AssertionError(f"{kind}: rho does not integrate back to Lambda")
    return rho, nu, d, label


def ct_rn(E: np.ndarray, kind: str) -> float:
    """C_T = integral rho dLambda = sum rho_k * delta_k."""
    rho, _nu, d, _ = rn_density(E, kind)
    return float(np.sum(rho * d))


# ---------------------------------------------------------------- existing family

def mus_existing(E: np.ndarray, kind: str, window: int = 5) -> np.ndarray:
    N = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    cos = N @ N.T
    n = len(E)
    out = []
    for k in range(n):
        if kind == "global":
            iu = np.triu_indices(n, k=1)
            mu = 1 - cos[iu].mean()
        elif kind == "window":
            s, e = max(0, k - window), min(n, k + window + 1)
            w = cos[s:e, s:e]
            iu = np.triu_indices_from(w, k=1)
            mu = 1 - w[iu].mean() if len(iu[0]) else 0.0
        elif kind == "narrow":
            s, e = max(0, k - 1), min(n, k + 2)
            w = cos[s:e, s:e]
            iu = np.triu_indices_from(w, k=1)
            mu = 1 - w[iu].mean() if len(iu[0]) else 0.0
        elif kind == "adjacent":
            a, b = max(0, k - 1), min(n - 1, k + 1)
            mu = 1 - cos[a, b] if a != b else 0.0
        elif kind == "none":
            mu = 1.0
        else:
            raise ValueError(kind)
        out.append(float(mu))
    return np.asarray(out)


def ct_existing(E: np.ndarray, kind: str) -> float:
    d = delta_of(E)
    m = mus_existing(E, kind)
    n = min(len(d), len(m))
    return float(np.sum(m[:n] * d[:n]))


def ct_of(E: np.ndarray, kind: str) -> float:
    return ct_rn(E, kind) if kind in RN else ct_existing(E, kind)


# ---------------------------------------------------------------- measurement

def sensitivity(E: np.ndarray, kind: str, rng) -> dict:
    asis = ct_of(E, kind)
    vals = np.asarray([ct_of(E[rng.permutation(len(E))], kind) for _ in range(NPERM)])
    mean = float(vals.mean())
    return {
        "ct": round(asis, 6),
        "shuf_mean": round(mean, 6),
        "bias": round(float((asis - mean) / mean), 4) if mean else 0.0,
        "abs_bias": round(abs(float((asis - mean) / mean)), 4) if mean else 0.0,
        "cv": round(float(vals.std() / mean), 4) if mean else 0.0,
        "pct": round(float((vals < asis).mean() * 100), 1),
    }


def identity_check() -> list[dict]:
    """The G1 anchor: every RN variant must satisfy sum(rho*nu) == sum(delta)."""
    rows = []
    for arm in ARMS:
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if not p.exists():
            continue
        E = np.load(p)
        for kind in RN:
            try:
                rho, nu, d, _ = rn_density(E, kind)
                err = abs(float(np.sum(rho * nu)) - float(np.sum(d)))
                rows.append({"arm": arm, "variant": kind, "ok": True, "absErr": err})
            except AssertionError as exc:
                rows.append({"arm": arm, "variant": kind, "ok": False, "absErr": None,
                             "note": str(exc)})
    return rows


def lebesgue_peek(arm: str) -> dict | None:
    """Connect to the math-tool layer: the framework's own dLambda = rho dnu + dM."""
    if lebesgue_decompose is None:
        return None
    p = CACHE / f"gpu_clsn_{arm}.npy"
    if not p.exists():
        return None
    E = np.load(p)
    d = delta_of(E)
    nu, _ = reference(E, "rn_uniform")
    m = min(len(d), len(nu))
    r = lebesgue_decompose(mu=[float(x) for x in d[:m]], nu=[float(x) for x in nu[:m]])
    return {k: r[k] for k in r if not isinstance(r[k], list)}


def main() -> int:
    rng = np.random.default_rng(5)
    out: dict = {}

    print("G1 · mu 的忠实离散化：RN 密度变体\n")
    print("=" * 78)
    print("第 0 步：密度恒等式核验（不是容差，是恒等式）")
    print("  sum_k rho_k * nu_k  必须等于  sum_k delta_k")
    print("=" * 78)
    idrows = identity_check()
    ok = sum(1 for r in idrows if r["ok"])
    worst = max((r["absErr"] or 0.0) for r in idrows) if idrows else 0.0
    print(f"  {ok}/{len(idrows)} 个 (arm, variant) 通过；最大绝对误差 {worst:.3e}")
    if ok != len(idrows):
        for r in idrows:
            if not r["ok"]:
                print(f"    FAIL {r['arm']} {r['variant']}: {r.get('note')}")
        print("\n  有变体不是密度，拒绝继续。")
        return 1

    print()
    print("=" * 78)
    print("第 1 步：顺序敏感度（as-filed 对 200 次随机置换）")
    print("=" * 78)
    print(f"  {'arm':<22}" + "".join(f"{k[:10]:>13}" for k in ALL))
    for arm in ARMS:
        p = CACHE / f"gpu_clsn_{arm}.npy"
        if not p.exists():
            continue
        E = np.load(p)
        rec = {}
        for kind in ALL:
            rec[kind] = sensitivity(E, kind, rng)
        out[arm] = rec
        print(f"  {arm:<22}" + "".join(f"{rec[k]['bias']:>13.3f}" for k in ALL))

    print()
    print("  平均 |bias| —— 越小越稳")
    means = {k: float(np.mean([abs(out[a][k]["bias"]) for a in out])) for k in ALL}
    for k in ALL:
        tag = "  ← 现行实现" if k == "window" else ("  ← μ≡1（此前最优）" if k == "none" else
              ("  ← G1 新变体" if k in RN else ""))
        print(f"    {k:<14} {means[k]:.4f}{tag}")
    best = min(means, key=means.get)
    print(f"\n  最稳：{best}  ({means[best]:.4f})")
    base = means["none"]
    beaten = [k for k in RN if means[k] < base]
    print(f"  比 μ≡1 更稳的 RN 变体：{beaten if beaten else '无'}")

    print()
    print("=" * 78)
    print("第 2 步：接到数学工具层（框架自己的 dΛ = ρ dν + dM）")
    print("=" * 78)
    peek = lebesgue_peek(ARMS[0])
    if peek is None:
        print("  （数学工具层不可用，跳过）")
    else:
        for k, v in peek.items():
            print(f"    {k:<24} {v}")

    out["_summary"] = {"mean_abs_bias": means, "best": best,
                       "rn_beats_none": beaten, "identity_ok": f"{ok}/{len(idrows)}",
                       "identity_max_abs_err": worst}
    f = M / "records_centroid" / "g1_rn_density.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
