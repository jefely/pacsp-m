"""Can each work's deposition path be described by a minimal curve family?

The pipeline already computes a per-snapshot profile -- delta_k and mu_k at every
piece -- then discards its shape by summing to the scalar C_T. The question here is
whether the cumulative profile

    C(k) = sum_{i<=k} mu_i * delta_i

has a simple functional form with few parameters, so that a work can be summarised by
that form instead of a single number.

Several candidate families are fitted to the empirically observed cumulative curves
and compared by R^2 and AIC, on the cumulative curve and on its normalised shape. The
normalised fit matters more: if works share a shape and differ only in scale, one
canonical curve plus an amplitude describes all of them.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import (  # noqa: E402
    compute_deltas, compute_embeddings, compute_mus, load_samples,
)

DATA = ROOT / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
WINDOW = 5

ARMS = ["poem", "machine_poem", "lyrics", "machine_lyrics", "techdoc",
        "machine_techdoc", "hc3_human_medicine", "hc3_ai_medicine",
        "hc3_human_openqa", "hc3_ai_openqa"]


# ---------------------------------------------------------------- candidates
def f_power(x, a, b):
    return a * np.power(x, b)


def f_exp(x, a, r):
    return a * (1.0 - np.exp(-r * x))


def f_logistic(x, a, r, x0):
    return a / (1.0 + np.exp(-r * (x - x0)))


def f_gompertz(x, a, r, x0):
    return a * np.exp(-np.exp(-r * (x - x0)))


def f_beta_cdf(x, a, b):
    # regularised incomplete beta, the natural 2-parameter family on [0,1]
    from scipy.special import betainc
    return betainc(a, b, np.clip(x, 1e-9, 1 - 1e-9))


def f_weibull(x, a, k, lam):
    return a * (1.0 - np.exp(-np.power(x / lam, k)))


CANDIDATES = [
    ("power      a*x^b", f_power, 2, False),
    ("exp        a(1-e^-rx)", f_exp, 2, False),
    ("logistic   3-param", f_logistic, 3, False),
    ("gompertz   3-param", f_gompertz, 3, False),
    ("weibull    3-param", f_weibull, 3, False),
    ("beta CDF   I_x(a,b)", f_beta_cdf, 2, True),
]


def fit(func, k, y, normalised):
    from scipy.optimize import curve_fit
    n = len(y)
    if normalised:
        x = (k - k[0]) / max(1e-9, (k[-1] - k[0]))
        target = y / y[-1] if y[-1] else y
    else:
        x = k.astype(float)
        target = y
    # initial guesses
    if func is f_power:
        p0 = [target[-1], 1.0]
    elif func is f_exp:
        p0 = [target[-1], 2.0]
    elif func is f_logistic:
        p0 = [target[-1], 4.0, x[len(x) // 2]]
    elif func is f_gompertz:
        p0 = [target[-1] * 1.05, 4.0, x[len(x) // 2]]
    elif func is f_weibull:
        p0 = [target[-1], 1.5, x[len(x) // 2]]
    else:
        p0 = [2.0, 2.0]
    bounds = ([-np.inf] * len(p0), [np.inf] * len(p0))
    if func is f_beta_cdf:
        bounds = ([1e-3, 1e-3], [60.0, 60.0])
    try:
        popt, _ = curve_fit(func, x, target, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        return None
    pred = func(x, *popt)
    resid = target - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    npar = len(popt)
    aic = n * np.log(max(ss_res / n, 1e-18)) + 2 * npar
    return {"r2": round(float(r2), 4), "aic": round(float(aic), 2),
            "params": [round(float(p), 4) for p in popt]}


def arm_curves(arm):
    p = DATA / arm
    if not p.is_dir():
        return None
    texts, files = load_samples(p)
    if len(texts) < 5:
        return None
    emb = compute_embeddings(texts, model_name=MODEL)
    d = np.asarray(compute_deltas(emb), dtype=float)
    m = np.asarray(compute_mus(emb, window=WINDOW), dtype=float)
    n = min(len(d), len(m))
    inc = m[:n] * d[:n]
    return np.cumsum(inc), inc


def main():
    report = {}
    for arm in ARMS:
        got = arm_curves(arm)
        if got is None:
            print(f"  [skip] {arm}")
            continue
        cum, inc = got
        k = np.arange(1, len(cum) + 1)
        print(f"\n=== {arm}  (T={len(cum)}, C_T={cum[-1]:.3f}) ===")
        print(f"  {'family':<24} {'R2 raw':>8} {'R2 norm':>8} {'params(norm)':>28}")
        row = {}
        for name, func, npar, _ in CANDIDATES:
            raw = fit(func, k, cum, False)
            nrm = fit(func, k, cum, True)
            if raw is None or nrm is None:
                print(f"  {name:<24}   fit failed")
                continue
            row[name] = {"raw": raw, "norm": nrm}
            print(f"  {name:<24} {raw['r2']:>8.4f} {nrm['r2']:>8.4f} "
                  f"{str(nrm['params']):>28}")
        report[arm] = row

    print()
    print("=== best family per corpus (normalised fit) ===")
    tally = {}
    for arm, row in report.items():
        if not row:
            continue
        best = max(row.items(), key=lambda kv: kv[1]["norm"]["r2"])
        tally[best[0]] = tally.get(best[0], 0) + 1
        print(f"  {arm:<22} {best[0]:<24} R2={best[1]['norm']['r2']:.4f} "
              f"params={best[1]['norm']['params']}")
    print()
    print(f"  family tally: {tally}")

    out = ROOT / "records_centroid" / "curve_fits.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n  written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
