"""Time each stage of a compare, so the slow part is identified rather than guessed at.

A run felt slow and the first instinct was the model. That is testable: measure import, model
construction, encoding, the distance matrices, and the bootstrap separately.
"""

import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
M = Path(r"D:\myproject\PACSP-M")
sys.path.insert(0, str(M))
sys.path.insert(0, str(M / "_ortgpu"))
os.environ["HF_HOME"] = str(M / "_hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"


def t(label, fn):
    t0 = time.time()
    r = fn()
    dt = time.time() - t0
    print(f"  {label:<44} {dt:>8.2f}s")
    return r, dt


def main():
    import numpy as np
    print("=== stage timings, poem vs machine_poem, 2000 bootstrap ===")

    import pacsp_tool as T
    _, t_import = t("import pacsp_tool", lambda: T)

    emb, t_mk = t("Embedder construction (lazy)", lambda: T.Embedder(
        "BAAI/bge-large-zh-v1.5", backend="onnx",
        onnx_dir=M / "onnx", prefer_gpu=True))

    ta, _ = T.load_texts(M / "data" / "poem")
    tb, _ = T.load_texts(M / "data" / "machine_poem")
    _, t_read = t("read corpora (62 files)", lambda: (ta, tb))

    EA, t_a = t(f"encode A ({len(ta)} texts, first call loads model)",
                lambda: emb.encode(ta))
    EB, t_b = t(f"encode B ({len(tb)} texts, cached model)",
                lambda: emb.encode(tb))
    print(f"      backend in use: {emb.active_backend} "
          f"{emb.describe().get('providers', '')}")

    wA, t_wa = t("within distances A", lambda: T.within_distances(EA))
    wB, t_wb = t("within distances B", lambda: T.within_distances(EB))
    X, t_x = t("cross distances 31x31", lambda: T.cross_distances(EA, EB))

    rng = np.random.default_rng(0)
    _, t_ma = t("bootstrap mean CI (2000) x2", lambda: (
        T.bootstrap_mean_ci(wA, 2000, rng), T.bootstrap_mean_ci(wB, 2000, rng)))

    def ratio_boot():
        out = []
        for _ in range(2000):
            ia = rng.integers(0, len(EA), len(EA))
            ib = rng.integers(0, len(EB), len(EB))
            out.append(T.within_distances(EA[ia]).mean() /
                       T.within_distances(EB[ib]).mean())
        return np.asarray(out)
    _, t_rb = t("ratio bootstrap (2000, recomputes distances)", ratio_boot)

    _, t_iv = t("cross-mean bootstrap (2000)", lambda: T.bootstrap_mean_ci(
        X, 2000, rng))
    _, t_st = t("style ratio (4000 pairs)", lambda: T.style_ratio(
        ta, tb, 4000, rng))
    _, t_as = t("assignability (LOO nearest centroid)", lambda:
                T.loo_assignability(EA, EB))

    total = t_import + t_mk + t_read + t_a + t_b + t_wa + t_wb + t_x + \
        t_ma + t_rb + t_iv + t_st + t_as
    print(f"\n  {'TOTAL':<44} {total:>8.2f}s")
    print("\n  ranking:")
    stages = [("encode A (model load)", t_a), ("ratio bootstrap", t_rb),
              ("encode B", t_b), ("bootstrap mean CI", t_ma),
              ("style ratio", t_st), ("cross-mean bootstrap", t_iv),
              ("import", t_import), ("cross distances", t_x),
              ("within A", t_wa), ("within B", t_wb),
              ("Embedder ctor", t_mk), ("assignability", t_as),
              ("read corpora", t_read)]
    for name, dt in sorted(stages, key=lambda x: -x[1])[:6]:
        print(f"    {name:<36} {dt:>7.2f}s  {100*dt/total:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
