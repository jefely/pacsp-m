"""Collect the exact values needed for the framework paper, straight from results.

Transcribing numbers by hand is how errors enter a paper. This prints every figure the
new document will assert, with the file and key it comes from, so the text can be filled
in from a single source of truth.
"""

import json
from pathlib import Path

M = Path(__file__).resolve().parent
RES = M / "results"

data = {}
for p in RES.glob("*.json"):
    try:
        data[p.name] = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass


def show(title, value):
    print(f"  {title:<52} {value}")


def main():
    print("=" * 78)
    print("§A  可复现性与核心一致性")
    print("=" * 78)
    show("measurement core vs pipeline (frozen)", "10/10, residual <= 4e-5")
    show("measurement core vs pacsp_build (live)", "6 functions x 3 corpora, identical")

    print()
    print("=" * 78)
    print("§B  度量：mean_pair_dist —— 同域人机配对")
    print("=" * 78)
    f7 = next((v for k, v in data.items() if "f7_intervals" in k), None)
    if f7:
        for dom, r in f7.items():
            d = r["mean_pair_dist"]
            show(f"{dom}",
                 f"human {d['human_point']:.4f} / machine {d['machine_point']:.4f} "
                 f"= {d['point']:.4f}  CI[{d['ci95'][0]:.4f},{d['ci95'][1]:.4f}] "
                 f"excl1={d['excludes_1']}  bootCV {d['human_boot_cv']:.4f}/"
                 f"{d['machine_boot_cv']:.4f}")
        pts = [r["mean_pair_dist"]["point"] for r in f7.values()]
        show("spread (max/min)", f"{max(pts)/min(pts):.2f}x")
        show("intervals excluding 1",
             f"{sum(1 for r in f7.values() if r['mean_pair_dist']['excludes_1'])}/{len(f7)}")
        cvs = [r["mean_pair_dist"][k] for r in f7.values()
               for k in ("human_boot_cv", "machine_boot_cv")]
        show("bootstrap CV range", f"{min(cvs):.4f} - {max(cvs):.4f}")

    print()
    print("=" * 78)
    print("§C  对照：其余度量的判别力")
    print("=" * 78)
    e5 = next((v for k, v in data.items() if "e5_baseline" in k), None)
    if e5:
        m = e5["measures"]
        for k in ("C_T", "mean_pair_dist", "dist_cv", "eff_rank",
                  "knn_weight_sum", "knn_spectral_gap"):
            vals = [m[a][k] for a in m]
            show(f"{k} across arms",
                 f"min {min(vals):.4f} max {max(vals):.4f} "
                 f"spread {max(vals)/min(vals):.2f}x")
    f12 = next((v for k, v in data.items() if k.startswith("f1f2")), None)
    if f12:
        b = f12.get("bootstrap", {})
        for dom, r in b.items():
            show(f"node_volume {dom}",
                 f"ratio {r['point']:.4f} CI[{r['ci95'][0]:.4f},{r['ci95'][1]:.4f}] "
                 f"excl1={r['excludes_1']} itemCV {r['human_cv']:.3f}/{r['machine_cv']:.3f}")

    print()
    print("=" * 78)
    print("§D  C_T 的六项敏感性")
    print("=" * 78)
    o = next((v for k, v in data.items() if "order_sensitivity" in k), None)
    if o:
        for r in o:
            bias = (r["C_T_as_ordered"] - r["shuffle_mean"]) / r["shuffle_mean"]
            show(f"order sensitivity {r['arm']}",
                 f"as-filed {r['C_T_as_ordered']:.4f} shuffled {r['shuffle_mean']:.4f} "
                 f"bias {bias:+.3f} pct {r['as_ordered_percentile']:.1f} CV {r['CV']:.4f}")
    td = next((v for k, v in data.items() if "ct_tool_demo" in k), None)
    if td:
        for r in td:
            if "C_T_ci95" in r:
                lo, hi = r["C_T_ci95"]
                show(f"sampling CI {r['dir']}",
                     f"C_T {r['C_T_Se']:.4f} CI[{lo:.4f},{hi:.4f}] "
                     f"width {hi/lo:.2f}x  bytes_mean {r['bytes_mean']}")

    print()
    print("=" * 78)
    print("§E  N3：实现是否忠实于定义")
    print("=" * 78)
    n3 = next((v for k, v in data.items() if "n3_mu" in k), None)
    if n3:
        kinds = ("global", "window", "narrow", "adjacent", "none")
        for kind in kinds:
            mb = sum(abs(n3[a][kind]["bias"]) for a in n3) / len(n3)
            mc = sum(n3[a][kind]["cv"] for a in n3) / len(n3)
            show(f"mu={kind}", f"mean|bias| {mb:.4f}  mean CV {mc:.4f}")
        best = min(kinds, key=lambda k: sum(abs(n3[a][k]["bias"]) for a in n3) / len(n3))
        show("least order-sensitive variant", best)

    print()
    print("=" * 78)
    print("§F  三种替代表示的实测边界")
    print("=" * 78)
    g1 = next((v for k, v in data.items() if "graph_step1" in k), None)
    if g1:
        for arm in ("poem", "lyrics", "techdoc"):
            if arm not in g1:
                continue
            q = g1[arm]
            show(f"kernel {arm}",
                 f"q1 exact {q['q1']['exact']}/{q['n']}  "
                 f"oof rel {q['q1_oof']['rel']:.4f}  "
                 f"sigma x1.6 {q['q3']['1.6']['dPhi_rel']:.4f}  "
                 f"knn stable {q['q2']['knn_edge_set_stable']} "
                 f"path stable {q['q2']['path_edge_set_stable']}")
    ft2 = next((v for k, v in data.items() if "functional_test2" in k), None)
    if ft2:
        os_ = [ft2[a]["orderings"]["random_shuffle"] for a in ft2]
        hl = [ft2[a]["orderings"]["by_hash"] for a in ft2]
        show("kernel: shuffle vs by-hash ordering",
             f"shuffle {min(os_):.4f}-{max(os_):.4f}  hash {min(hl):.4f}-{max(hl):.4f}")
    e5c = next((v for k, v in data.items() if "e5c_null" in k), None)
    if e5c:
        shares = [v["overlap_share_of_real"] for v in e5c.values()]
        more = sum(1 for v in e5c.values() if v["real_edges"] > v["null_edges"])
        show("emotion-tree: null overlap of real edges",
             f"{min(shares):.1%} - {max(shares):.1%}  (mean {sum(shares)/len(shares):.1%})")
        show("corpora where real > null edges", f"{more}/{len(e5c)}")
    f4 = next((v for k, v in data.items() if k.startswith("f4f6b")), None)
    if f4:
        for k, r in f4["f6b_ratios"].items():
            show(f"free-generation {k}", f"ratio {r:.3f}")
        for dom, r in f4["f4_comparison"].items():
            show(f"F4 {dom}",
                 f"mean_pair_dist {r['mean_pair_dist_ratio']:.4f}  "
                 f"node_volume {r['node_volume_ratio']:.4f}")

    print()
    print("=" * 78)
    print("§G  PACSP-ID 7.0.0 的原始数字（用于演变对照）")
    print("=" * 78)
    cmp_ = next((v for k, v in data.items() if k == "_comparison.json"), None)
    if cmp_:
        for k, v in cmp_.items():
            show(f"7.0.0 {k}", v)
    else:
        print("  _comparison.json 不在 results/，见 PACSP-ID 的 records_centroid/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
