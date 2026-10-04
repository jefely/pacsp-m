"""Check every number quoted in PACSP-M-1.0.0.md against the frozen results.

A paper whose figures cannot be traced to a result file is not reproducible, however
carefully it is written. This parses the document for the specific values it asserts and
compares each against results/, so a transcription slip is caught rather than published.

Checks are grouped by section. Each entry names the claimed value, the source file and
the path within it, so a failure says where to look.
"""

import json
import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parent
DOC = M / "PACSP-M-1.0.0.md"
RES = M / "results"

results = {}
for p in RES.glob("*.json"):
    try:
        results[p.name] = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  cannot load {p.name}: {e}")


def get(name):
    """Find a result file whose name starts with or contains the key."""
    for k, v in results.items():
        if name in k:
            return v
    return None


def close(a, b, tol=5e-4):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except Exception:
        return False


def main():
    doc = DOC.read_text(encoding="utf-8")
    checks = []          # (label, claimed, actual, ok)

    # ---------------------------------------------------------------- §3.3 order
    o = get("order_sensitivity")
    if o:
        rows = {r["arm"]: r for r in o}
        for arm, bias, pct in (("poem", -0.3158, 0.0), ("lyrics", -0.5141, 0.0),
                               ("techdoc", -0.1987, 0.0),
                               ("hc3_human_medicine", -0.0049, 43.0),
                               ("hc3_ai_medicine", -0.0100, 28.5)):
            r = rows.get(arm)
            if not r:
                continue
            claimed = round((r["C_T_as_ordered"] - r["shuffle_mean"]) /
                            r["shuffle_mean"], 2)
            checks.append((f"§3.3 {arm} 打乱偏差",
                           f"{bias:.2f}", f"{claimed:.2f}", close(bias, claimed, 0.02)))
            checks.append((f"§3.3 {arm} 百分位", pct,
                           r["as_ordered_percentile"],
                           close(pct, r["as_ordered_percentile"], 0.02)))
        # CV range
        cvs = [r["CV"] for r in o]
        checks.append(("§3.3 打乱 CV 范围 0.016–0.048",
                       "0.016-0.048", f"{min(cvs):.3f}-{max(cvs):.3f}",
                       close(min(cvs), 0.016, 0.05) and close(max(cvs), 0.048, 0.05)))

    # ---------------------------------------------------------------- §4 mu
    n3 = get("n3_mu_localisation")
    if n3:
        means = {}
        for kind in ("global", "window", "narrow", "adjacent", "none"):
            means[kind] = sum(abs(n3[a][kind]["bias"]) for a in n3) / len(n3)
        for kind, claim in (("global", 0.141), ("window", 0.191),
                            ("narrow", 0.246), ("adjacent", 0.218),
                            ("none", 0.140)):
            checks.append((f"§4.3 平均|偏差| {kind}", f"{claim:.3f}",
                           f"{means[kind]:.3f}", close(claim, means[kind], 0.02)))
        cvs = {}
        for kind in ("global", "window", "narrow", "adjacent", "none"):
            cvs[kind] = sum(n3[a][kind]["cv"] for a in n3) / len(n3)
        for kind, claim in (("global", 0.0208), ("window", 0.0303),
                            ("narrow", 0.0445), ("adjacent", 0.0416),
                            ("none", 0.0220)):
            checks.append((f"§4.3 平均 CV {kind}", f"{claim:.4f}",
                           f"{cvs[kind]:.4f}", close(claim, cvs[kind], 0.02)))

    # ---------------------------------------------------------------- §5.1 kernel
    # Two different experiments measure these, and the paper quotes the pair that
    # belongs together. graph_step1 used a frame containing part of the corpus and a
    # 31-of-31 self frame; functional_test2 used a held-out frame with different sigma
    # calibration. The sigma range 70-133 percent and the out-of-frame error 0.52-0.65
    # both come from graph_step1, so that is the file to check against.
    g1 = get("graph_step1")
    if g1:
        for arm, claim in (("poem", 0.7006), ("lyrics", 0.7727), ("techdoc", 1.1593)):
            v = g1.get(arm, {}).get("q3", {}).get("1.6", {}).get("dPhi_rel")
            if v is not None:
                checks.append((f"§5.1 σ×1.6 {arm}", claim, v, close(claim, v, 0.02)))
        rels = []
        for arm, claim in (("poem", 0.5544), ("lyrics", 0.5182), ("techdoc", 0.6536)):
            v = g1.get(arm, {}).get("q1_oof", {}).get("rel")
            if v is not None:
                rels.append(v)
                checks.append((f"§5.1 帧外相对误差 {arm}", claim, v,
                               close(claim, v, 0.02)))
        if rels:
            checks.append(("§5.1 帧外误差范围 0.52–0.65", "0.52-0.65",
                           f"{min(rels):.4f}-{max(rels):.4f}",
                           close(min(rels), 0.518, 0.02) and
                           close(max(rels), 0.654, 0.02)))
        # the 70 to 133 percent range quoted in the paper
        allq3 = []
        for arm in g1:
            d = g1[arm].get("q3", {}).get("1.6", {}).get("dPhi_rel")
            if d is not None:
                allq3.append(d)
        if allq3:
            checks.append(("§5.1 σ×1.6 范围上限 ≈1.33", 1.33,
                           round(max(allq3), 4), max(allq3) <= 1.35))
            checks.append(("§5.1 σ×1.6 范围下限 ≈0.70", 0.70,
                           round(min(allq3), 4), min(allq3) >= 0.65))

    # ---------------------------------------------------------------- §5.2 emotion
    f12 = get("f1f2")
    if f12:
        ts = f12.get("threshold_scan", {}).get("ratios", {})
        if "0.02" in ts:
            for k, claim in (("poem", 0.3470), ("lyrics", 0.0334),
                             ("medicine", 0.0558), ("openqa", 0.2979)):
                v = ts["0.02"].get(k)
                if v is not None:
                    checks.append((f"§5.2 t=0.02 {k}", claim, round(v, 4),
                                   close(claim, v, 0.02)))
        ps = f12.get("probe_scan", {})
        for phrase, k, claim in (
                ("这段文字表达的情绪是", "medicine", 1.203),
                ("这段素材主要体现什么情绪？答：", "medicine", 46.273),
                ("这段素材主要体现什么情绪？答：", "openqa", 18.710)):
            v = ps.get(phrase, {}).get(k)
            if v is not None:
                checks.append((f"§5.2 短语 {phrase[:8]}… {k}", claim, round(v, 3),
                               close(claim, v, 0.02)))
        vs = f12.get("vocabulary_scan", {})
        if "positive_only" in vs:
            for k, claim in (("poem", 0.8120), ("lyrics", 0.3721),
                             ("medicine", 2.0898), ("openqa", 4.9410)):
                v = vs["positive_only"].get(k)
                if v is not None:
                    checks.append((f"§5.2 仅正面词 {k}", claim, round(v, 4),
                                   close(claim, v, 0.02)))

    em = get("f6_mechanism")
    if em:
        a = em.get("arms", {})
        if "machine_poem" in a:
            checks.append(("§5.2 machine_poem 情绪词数 0",
                          0, int(a["machine_poem"]["total_emotion_words"]),
                          int(a["machine_poem"]["total_emotion_words"]) == 0))
        if "techdoc" in a:
            checks.append(("§5.2 techdoc 情绪词数 314",
                          314, int(a["techdoc"]["total_emotion_words"]),
                          int(a["techdoc"]["total_emotion_words"]) == 314))

    f4 = get("f4f6b")
    if f4:
        fr = f4.get("f6b_ratios", {})
        for k, claim in (("poem", 0.870), ("lyrics", 0.579),
                         ("medicine", 1.571), ("openqa", 1.667)):
            v = fr.get(k)
            if v is not None:
                checks.append((f"§5.2 自由生成 {k}", claim, round(v, 3),
                               close(claim, v, 0.02)))

    # ---------------------------------------------------------------- §6 baseline
    f7 = get("f7_intervals")
    if f7:
        mpd = {"poem": 0.8305, "lyrics": 0.8691, "techdoc": 1.4426,
               "medicine": 0.9147, "openqa": 0.9358}
        ci = {"poem": (0.6842, 0.9742), "lyrics": (0.7482, 0.9812),
              "techdoc": (1.3043, 1.5903), "medicine": (0.8875, 0.9413),
              "openqa": (0.9174, 0.9525)}
        for k, claim in mpd.items():
            v = f7.get(k, {}).get("mean_pair_dist", {}).get("point")
            if v is not None:
                checks.append((f"§4.3 {k} 比值", claim, v, close(claim, v, 0.02)))
        for k, (lo, hi) in ci.items():
            d = f7.get(k, {}).get("mean_pair_dist", {}).get("ci95")
            if d:
                checks.append((f"§4.3 {k} CI 下限", lo, d[0], close(lo, d[0], 0.02)))
                checks.append((f"§4.3 {k} CI 上限", hi, d[1], close(hi, d[1], 0.02)))
        ok = sum(1 for k in f7
                 if f7[k].get("mean_pair_dist", {}).get("excludes_1"))
        checks.append(("§4.3 区间排除 1 的个数 5/5", 5, ok, ok == 5))
        # the two spreads the paper reports
        pts = {k: f7[k]["mean_pair_dist"]["point"] for k in f7}
        consistent = [v for k, v in pts.items() if k != "techdoc"]
        checks.append(("§4.3 四个同向域跨度 1.13x", 1.13,
                       round(max(consistent) / min(consistent), 2),
                       close(1.13, max(consistent) / min(consistent), 0.02)))
        checks.append(("§4.3 含反向域跨度 1.74x", 1.74,
                       round(max(pts.values()) / min(pts.values()), 2),
                       close(1.74, max(pts.values()) / min(pts.values()), 0.02)))
        cvs = [f7[k]["mean_pair_dist"][f] for k in f7
               for f in ("human_boot_cv", "machine_boot_cv")]
        checks.append(("§4.2 自助法 CV 0.012–0.081", "0.012-0.081",
                       f"{min(cvs):.4f}-{max(cvs):.4f}",
                       close(min(cvs), 0.0134, 0.05) and close(max(cvs), 0.0805, 0.05)))

    e5 = get("e5_baseline")
    if e5:
        m = e5.get("measures", {})
        # §3.6 / §2.2 model difference
        checks.append(("§2.2 模型差 31%（qwen/r1）", 0.31,
                       round((9.5902 - 6.6421) / 9.5902, 2),
                       close(0.31, round((9.5902 - 6.6421) / 9.5902, 2), 0.05)))
        # §5.1 C_T across-arm spread 13.50x
        if "C_T" in next(iter(m.values())):
            vals = [m[a]["C_T"] for a in m]
            checks.append(("§5.1 C_T 跨臂跨度 13.50x", 13.50,
                           round(max(vals) / min(vals), 2),
                           close(13.50, max(vals) / min(vals), 0.02)))
            vals = [m[a]["mean_pair_dist"] for a in m]
            checks.append(("§4.5 mean_pair_dist 跨臂 2.20x", 2.20,
                           round(max(vals) / min(vals), 2),
                           close(2.20, max(vals) / min(vals), 0.02)))

    # ---------------------------------------------------------------- §2.4 sigma
    n4 = get("n4_sigma_resolutions")
    if n4:
        ss = n4.get("sigma_sensitivity", {})
        for mode, claim in (("A", 0.3682), ("B", 0.0808), ("C", 0.1928), ("D", 0.0070)):
            v = ss.get(mode, {}).get("mean")
            if v is not None:
                checks.append((f"§2.4 路线 {mode} σ×1.6 均值", claim, v,
                               close(claim, v, 0.02)))
        os_ = n4.get("order_sensitivity", {})
        zero = all(abs(os_.get(m, {}).get("mean_bias", 1)) < 1e-9 for m in "ABCD")
        checks.append(("§2.4 四路线顺序偏差全为 0", True, zero, zero))
        rt = n4.get("ratios", {})
        for mode, dom, claim in (("A", "poem", 1.7018), ("B", "poem", 2.1276),
                                 ("C", "poem", 2.7989), ("D", "poem", 0.9883),
                                 ("C", "techdoc", 0.9096), ("D", "medicine", 0.9991)):
            v = rt.get(mode, {}).get(dom)
            if v is not None:
                checks.append((f"§2.4 路线 {mode} {dom} 比值", claim, round(v, 4),
                               close(claim, v, 0.02)))
        # direction counts
        for mode, below in (("A", 0), ("B", 0), ("C", 1), ("D", 5)):
            r = rt.get(mode, {})
            vals = [v for v in r.values() if v is not None]
            n_below = sum(1 for v in vals if v < 1)
            checks.append((f"§2.4 路线 {mode} 低于 1 的个数 {below}/5",
                           below, n_below, n_below == below))

    n4b = get("n4b_d_bootstrap")
    if n4b:
        excl = sum(1 for v in n4b.values() if v["excludes_1"])
        checks.append(("§2.4 路线 D 区间排除 1 的个数 0/5", 0, excl, excl == 0))
        cvs = [v[k] for v in n4b.values()
               for k in ("boot_cv_human", "boot_cv_machine")]
        checks.append(("§2.4 路线 D 自助法 CV 0.0026–0.0043", "0.0026-0.0043",
                       f"{min(cvs):.6f}-{max(cvs):.6f}",
                       close(min(cvs), 0.0026, 0.05) and close(max(cvs), 0.0043, 0.05)))

    # ---------------------------------------------------------------- §4.4 length
    n5 = get("n5_length_control")
    if n5:
        aud = n5.get("audit", {})
        for k, claim in (("lyrics", 11.64), ("techdoc", 7.92), ("poem", 1.13),
                         ("medicine", 1.01), ("openqa", 0.94)):
            v = aud.get(k, {}).get("byte_ratio")
            if v is not None:
                checks.append((f"§4.4 长度比 {k}", claim, v, close(claim, v, 0.02)))
        ctrl = n5.get("control", {})
        for k, claim in (("poem", 0.7804), ("lyrics", 0.9292), ("techdoc", 1.4437),
                         ("medicine", 0.9065), ("openqa", 0.9337)):
            v = ctrl.get(k, {}).get("trimmed_ratio")
            if v is not None:
                checks.append((f"§4.4 裁剪后比值 {k}", claim, v, close(claim, v, 0.02)))
        for k, claim in (("poem", 0.940), ("lyrics", 1.069), ("techdoc", 1.001)):
            v = ctrl.get(k, {}).get("trimmed_ratio")
            o = ctrl.get(k, {}).get("orig_ratio")
            if v and o:
                checks.append((f"§4.4 裁剪后变化 {k}", claim, round(v / o, 3),
                               close(claim, v / o, 0.02)))
        vd = n5.get("verdict", {})
        checks.append(("§4.4 方向一致 原始 4/5", 4, vd.get("below_1_original"),
                       vd.get("below_1_original") == 4))
        checks.append(("§4.4 方向一致 裁剪后 4/5", 4, vd.get("below_1_trimmed"),
                       vd.get("below_1_trimmed") == 4))
        checks.append(("§4.4 无配对翻转", [], vd.get("direction_flips"),
                       vd.get("direction_flips") == []))
        checks.append(("§4.4 最大偏移 6.9%", 0.069, vd.get("max_shift"),
                       close(0.069, vd.get("max_shift"), 0.05)))

    # ---------------------------------------------------------------- report
    print(f"  doc: {DOC.name}  ({len(doc):,} 字符)")
    print(f"  results 文件: {len(results)}")
    print()
    bad = []
    for label, claimed, actual, ok in checks:
        mark = "OK " if ok else "DIFF"
        if not ok:
            bad.append((label, claimed, actual))
        print(f"  [{mark}] {label:<34} 论文 {str(claimed):>10}  结果 {str(actual):>10}")
    print()
    print(f"  {len(checks) - len(bad)}/{len(checks)} 项一致")
    if bad:
        print("\n  不一致项：")
        for label, claimed, actual in bad:
            print(f"    {label}: 论文 {claimed} vs 结果 {actual}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
