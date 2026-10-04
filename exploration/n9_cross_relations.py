"""N9: can the frame calibrate relations between collections, not just within one?

Everything reported so far reduces a collection to one number, D. The question is whether
the frame supports relations between collections: an approximate interval for how far
apart two groups are, a style-similarity notion, and a shared region.

All three are computable from the cross-distance matrix, which is the n-by-m matrix of
distances between every text of A and every text of B. Unlike D, this is not reduced to a
scalar at the source, so the three relations can be read off it separately.

  interval   distribution of the n*m cross distances, with a bootstrap interval on its mean
  style      within-A and within-B mean distances against the cross mean. If cross is close
             to both within values the two groups are not separated; the ratio of cross to
             the pooled within distance is the separation statistic
  overlap    fraction of cross distances that fall below the pooled within-set 95th
             percentile, which is the share of B that sits inside A's own radius

A caution that determines how these may be read: N7 found the human-machine difference is
not a templating difference, so a stylometric measure would not be expected to track it.
That is checked rather than assumed, because if it does track it, section 7.1's caveat gets
stronger.
"""

import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"
CACHE = M / "_hf_home"
NBOOT = 4000
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", "machine_techdoc2"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]

FUNC = set("""的 了 是 在 和 与 也 就 都 而 及 或 一个 我们 你们 他们 这 那 有 不 人 上 下
中 为 以 到 说 要 会 对 能 可以 因为 所以 但是 如果 虽然 然后 这样 那样 什么 怎么 就是
已经 还是 只是 没有 自己 这个 那个 一些 一样 时候 现在 知道 觉得 应该 可能 需要 进行
通过 对于 关于 根据 由于 以及 并且 或者 例如 因此 然而 不过 而且""".split())


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    return np.asarray(m.encode(texts, batch_size=8), dtype=np.float64)


def within(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return np.sqrt(sq(E, E)[iu])


def cross(EA, EB):
    return np.sqrt(sq(EA, EB)).ravel()


def jaccard(a, b):
    ga = set(a[i:i + 2] for i in range(max(1, len(a) - 1)))
    gb = set(b[i:i + 2] for i in range(max(1, len(b) - 1)))
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def main():
    import os
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")

    rng = np.random.default_rng(41)
    cache = {}
    for h, m in PAIRS:
        for arm in (h, m):
            if arm not in cache and (DATA / arm).is_dir():
                cache[arm] = pacsp_core.load_samples(DATA / arm)[0]

    out = {"pairs": {}}
    print(f"  {'pair':<12} {'cross mean':>11} {'95% CI':>30} {'within A':>9} "
          f"{'within B':>9} {'sep':>7} {'overlap':>8}")
    for h, m in PAIRS:
        if h not in cache or m not in cache:
            continue
        EA, EB = embed(cache[h]), embed(cache[m])
        wA, wB = within(EA), within(EB)
        X = cross(EA, EB)

        # interval on the mean cross distance
        boots = np.array([rng.choice(X, len(X), replace=True).mean()
                          for _ in range(NBOOT)])
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

        pooled = np.concatenate([wA, wB])
        sep = float(X.mean() / pooled.mean())
        # share of B lying inside A's own radius: pooled within 95th percentile
        thr = float(np.percentile(pooled, 95))
        overlap = float((X < thr).mean())

        dom = h.split("_")[-1]
        out["pairs"][dom] = {
            "cross_mean": round(float(X.mean()), 4),
            "cross_ci95": [round(ci[0], 4), round(ci[1], 4)],
            "cross_sd": round(float(X.std()), 4),
            "within_human_mean": round(float(wA.mean()), 4),
            "within_machine_mean": round(float(wB.mean()), 4),
            "separation": round(sep, 4),
            "overlap_at_within_p95": round(overlap, 4),
            "n_cross": int(len(X)),
        }
        print(f"  {dom:<12} {X.mean():>11.4f} "
              f"{'[' + format(ci[0], '.4f') + ', ' + format(ci[1], '.4f') + ']':>30} "
              f"{wA.mean():>9.4f} {wB.mean():>9.4f} {sep:>7.4f} {overlap:>8.3f}")

    # ---------------------------------------------------------------- style probe
    print()
    print("=== does a lexical style measure track the same separation? ===")
    print("  (if it does, D is partly style; if not, the two are independent)")
    print(f"  {'pair':<12} {'D ratio':>9} {'style ratio':>12} {'P95 overlap':>12}")
    f7 = None
    for p in (M / "results").glob("f7_intervals*.json"):
        f7 = json.loads(p.read_text(encoding="utf-8"))
    for h, m in PAIRS:
        if h not in cache or m not in cache:
            continue
        dom = h.split("_")[-1]
        jw = [jaccard(a, b) for i, a in enumerate(cache[h]) for b in cache[h][i + 1:]]
        jc = [jaccard(a, b) for a in cache[h] for b in cache[m]]
        sratio = float(np.mean(jc) / np.mean(jw)) if jw else float("nan")
        d = f7.get(dom, {}).get("mean_pair_dist", {}).get("point") if f7 else None
        ov = out["pairs"][dom]["overlap_at_within_p95"]
        print(f"  {dom:<12} {(d if d else float('nan')):>9.4f} {sratio:>12.4f} "
              f"{ov:>12.3f}")
        out["pairs"][dom]["style_ratio_cross_over_within"] = round(sratio, 4)

    # ---------------------------------------------------------------- reading
    print()
    print("=== reading ===")
    seps = [v["separation"] for v in out["pairs"].values()]
    ovs = [v["overlap_at_within_p95"] for v in out["pairs"].values()]
    print(f"  separation (cross / pooled within): "
          f"{min(seps):.4f} - {max(seps):.4f}")
    print(f"  overlap at pooled within P95      : "
          f"{min(ovs):.3f} - {max(ovs):.3f}")
    print(f"  pairs with separation > 1 (groups apart): "
          f"{sum(1 for s in seps if s > 1)}/{len(seps)}")
    print(f"  pairs with separation < 1 (groups interpenetrate): "
          f"{sum(1 for s in seps if s < 1)}/{len(seps)}")
    out["summary"] = {
        "separation_range": [round(min(seps), 4), round(max(seps), 4)],
        "overlap_range": [round(min(ovs), 4), round(max(ovs), 4)],
        "n_sep_gt_1": sum(1 for s in seps if s > 1),
        "n_sep_lt_1": sum(1 for s in seps if s < 1),
    }

    # correlation between D ratio and style ratio
    ds, ss = [], []
    for dom, v in out["pairs"].items():
        if f7 and dom in f7:
            ds.append(f7[dom]["mean_pair_dist"]["point"])
            ss.append(v["style_ratio_cross_over_within"])
    if len(ds) >= 3:
        r = float(np.corrcoef(ds, ss)[0, 1])
        print(f"\n  correlation of D ratio with style ratio: r = {r:.4f} (n={len(ds)})")
        out["summary"]["r_D_vs_style"] = round(r, 4)

    f = M / "records_centroid" / f"n9_cross_relations_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
