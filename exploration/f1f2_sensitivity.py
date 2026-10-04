"""F1 and F2: sensitivity of the emotion-tree features, and intervals on the ratios.

F1 varies the three free parameters. The probe is run once per corpus and the resulting
probability matrix Y is cached, so the threshold and vocabulary scans are pure
post-processing and cost nothing; only the probe phrase needs the model again.

    threshold t      re-derive the hierarchy from the cached Y
    vocabulary       restrict Y to subsets by probability mass and by word class
    probe phrase     re-probe with alternative phrasings

F2 puts an interval on the human-versus-machine ratio. Each corpus supplies 31 items, so
the items can be resampled with replacement and node_volume recomputed, giving a
distribution for each corpus and hence for the ratio. This is the check that was missing
when the ratios were first reported as bare numbers.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pacsp_core import load_samples  # noqa: E402

DATA = ROOT / "data"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
LM = "Qwen/Qwen2.5-7B-Instruct"
LIMIT = 31
CACHE = ROOT / "records_centroid" / "probe_cache"
NBOOT = 2000

PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", None), ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]

PROBES = [
    "这段素材的核心情绪是",
    "这段文字表达的情绪是",
    "这段素材主要体现什么情绪？答：",
    "情绪：",
]

from emotion_tree_probe import (  # noqa: E402
    EMOTIONS, build_hierarchy, emotion_token_ids, load_model, probe_matrix,
)

# coarse groupings, used to test whether the vocabulary choice matters
POSITIVE = {"喜悦", "快乐", "高兴", "兴奋", "欣喜", "愉快", "满足", "幸福", "欣慰",
            "感动", "乐观", "希望", "期待", "憧憬", "向往", "自信", "骄傲", "自豪",
            "得意", "轻松", "平静", "安宁", "宁静", "放松", "释然", "坦然", "安心",
            "舒畅", "惬意", "从容", "爱", "喜爱", "热爱", "眷恋", "思念", "怀念",
            "牵挂", "温柔", "亲切", "亲密", "感激", "感谢", "敬佩", "尊敬", "崇拜",
            "仰慕", "信任", "依赖", "认同", "归属", "鼓励", "激励", "振奋", "鼓舞",
            "坚定", "决心", "勇气", "坚韧", "执着", "毅力"}
NEGATIVE = {"悲伤", "难过", "伤心", "痛苦", "悲痛", "哀伤", "凄凉", "忧郁", "沮丧",
            "失落", "孤独", "寂寞", "空虚", "惆怅", "迷茫", "困惑", "无奈", "无力",
            "绝望", "消沉", "愤怒", "生气", "恼怒", "不满", "怨恨", "厌恶", "反感",
            "嫉妒", "轻蔑", "敌意", "恐惧", "害怕", "惊恐", "焦虑", "紧张", "担忧",
            "不安", "惶恐", "畏惧", "战栗", "羞愧", "内疚", "懊悔", "自责", "尴尬",
            "羞耻", "负罪", "后悔", "遗憾", "惋惜", "厌倦", "疲惫", "倦怠"}


def cache_path(arm, probe_idx):
    return CACHE / f"{arm}__p{probe_idx}.npy"


def get_Y(arm, probe_idx, tok, mdl, torch, ids, words):
    """Probe once and cache, keyed by corpus and probe phrase."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = cache_path(arm, probe_idx)
    if p.exists():
        return np.load(p)
    texts, _ = load_samples(DATA / arm)
    texts = texts[:LIMIT]
    import emotion_tree_probe as etp
    saved = etp.PROBE
    etp.PROBE = PROBES[probe_idx]
    try:
        Y, _, _ = probe_matrix(texts, tok, mdl, torch, ids)
    finally:
        etp.PROBE = saved
    np.save(p, Y)
    return Y


def hierarchy_from_Y(Y, words, t, keep_idx=None):
    """Rebuild the graph on a (possibly restricted) vocabulary."""
    if keep_idx is not None:
        Y = Y[:, keep_idx]
        words = [words[i] for i in keep_idx]
    C = Y.T @ Y
    edges, rs, cs = build_hierarchy(C, t=t)
    return edges, words, C


def edge_volume(edges):
    return float(sum(w for _, _, w in edges))


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "_hf_home"))
    os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "_hf_home" / "hub"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    arms = [a for pair in PAIRS for a in pair if a]
    tok, mdl, torch = load_model(LM)
    ids = emotion_token_ids(tok, EMOTIONS)
    words = list(ids.keys())
    print(f"  vocab: {len(words)} single-token emotion words")

    print("\n=== probing (cached) ===")
    t0 = time.time()
    Y = {}
    for a in arms:
        Y[a] = get_Y(a, 0, tok, mdl, torch, ids, words)
    print(f"  {len(arms)} corpora probed in {time.time()-t0:.1f}s")

    out = {"vocab_size": len(words)}

    # ---------------------------------------------------------------- F1a threshold
    print("\n=== F1a: threshold sensitivity ===")
    print(f"  {'t':>8} " + "".join(f"{a[:11]:>13}" for a in arms))
    thr = {}
    for t in (0.005, 0.01, 0.02, 0.05, 0.10):
        row = {}
        for a in arms:
            e, _, _ = hierarchy_from_Y(Y[a], words, t)
            row[a] = edge_volume(e)
        thr[str(t)] = row
        print(f"  {t:>8} " + "".join(f"{row[a]:>13.6f}" for a in arms))
    # human/machine ratios per threshold
    print(f"\n  {'t':>8} {'poem':>10} {'lyrics':>10} {'medicine':>10} {'openqa':>10}")
    ratios = {}
    for t, row in thr.items():
        r = {}
        for h, m in PAIRS:
            if m:
                r[f"{h.split('_')[-1]}"] = row[h] / row[m] if row[m] else float("nan")
        ratios[t] = r
        print(f"  {t:>8} " + "".join(
            f"{r.get(k, float('nan')):>10.4f}"
            for k in ("poem", "lyrics", "medicine", "openqa")))
    out["threshold_scan"] = {"edge_volume": thr, "ratios": ratios}

    # ---------------------------------------------------------------- F1b vocabulary
    print("\n=== F1b: vocabulary sensitivity (ratio edge_volume human/machine) ===")
    mass = Y[arms[0]].sum(axis=0) + Y[arms[1]].sum(axis=0)
    order = np.argsort(mass)[::-1]
    voc_scan = {}
    variants = {
        "all": None,
        "top70": order[:70],
        "top40": order[:40],
        "top20": order[:20],
        "positive_only": np.array([i for i, w in enumerate(words) if w in POSITIVE]),
        "negative_only": np.array([i for i, w in enumerate(words) if w in NEGATIVE]),
    }
    print(f"  {'variant':<16} {'n':>5} " + "".join(
        f"{k:>10}" for k in ("poem", "lyrics", "medicine", "openqa")))
    for name, keep in variants.items():
        row = {}
        for h, m in PAIRS:
            if not m:
                continue
            eh, _, _ = hierarchy_from_Y(Y[h], words, 0.02, keep)
            em, _, _ = hierarchy_from_Y(Y[m], words, 0.02, keep)
            vh, vm = edge_volume(eh), edge_volume(em)
            row[h.split("_")[-1]] = vh / vm if vm else float("nan")
        voc_scan[name] = {k: round(v, 6) for k, v in row.items()}
        n = len(keep) if keep is not None else len(words)
        print(f"  {name:<16} {n:>5} " + "".join(
            f"{row.get(k, float('nan')):>10.4f}"
            for k in ("poem", "lyrics", "medicine", "openqa")))
    out["vocabulary_scan"] = voc_scan

    # ---------------------------------------------------------------- F1c probe phrase
    print("\n=== F1c: probe phrase sensitivity ===")
    probe_scan = {}
    for pi, phrase in enumerate(PROBES):
        if pi == 0:
            continue
        print(f"  probing with {phrase!r}")
        row = {}
        for h, m in PAIRS:
            if not m:
                continue
            Yh = get_Y(h, pi, tok, mdl, torch, ids, words)
            Ym = get_Y(m, pi, tok, mdl, torch, ids, words)
            eh, _, _ = hierarchy_from_Y(Yh, words, 0.02)
            em, _, _ = hierarchy_from_Y(Ym, words, 0.02)
            vh, vm = edge_volume(eh), edge_volume(em)
            row[h.split("_")[-1]] = vh / vm if vm else float("nan")
        probe_scan[phrase] = {k: round(v, 6) for k, v in row.items()}
        print(f"    " + "".join(f"{k} {row.get(k, float('nan')):.4f}  "
                                for k in ("poem", "lyrics", "medicine", "openqa")))
    out["probe_scan"] = probe_scan

    # ---------------------------------------------------------------- F2 bootstrap
    print("\n=== F2: bootstrap intervals on node_volume and the ratio ===")
    rng = np.random.default_rng(7)
    boot = {}
    for h, m in PAIRS:
        if not m:
            continue
        key = h.split("_")[-1]
        nv = {}
        for arm in (h, m):
            # node_volume is the total probe mass, so resampling items resamples it
            item_tot = Y[arm].sum(axis=1)
            idx = rng.integers(0, len(item_tot), size=(NBOOT, len(item_tot)))
            samples = item_tot[idx].mean(axis=1) * len(item_tot)
            nv[arm] = samples
        ratio = nv[h] / nv[m]
        lo, hi = np.percentile(ratio, [2.5, 97.5])
        point = Y[h].sum() / Y[m].sum()
        boot[key] = {
            "point": round(float(point), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "excludes_1": bool(lo > 1 or hi < 1),
            "human_mean": round(float(Y[h].sum(axis=1).mean()), 4),
            "human_cv": round(float(Y[h].sum(axis=1).std() /
                                    Y[h].sum(axis=1).mean()), 4),
            "machine_mean": round(float(Y[m].sum(axis=1).mean()), 4),
            "machine_cv": round(float(Y[m].sum(axis=1).std() /
                                     Y[m].sum(axis=1).mean()), 4),
        }
        b = boot[key]
        print(f"  {key:<10} ratio {b['point']:.4f}  95% CI "
              f"[{b['ci95'][0]:.4f}, {b['ci95'][1]:.4f}]  "
              f"excludes 1: {b['excludes_1']}   "
              f"(human CV {b['human_cv']:.3f}, machine CV {b['machine_cv']:.3f})")
    out["bootstrap"] = boot

    f = ROOT / "records_centroid" / f"f1f2_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
