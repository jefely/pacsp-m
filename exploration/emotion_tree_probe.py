"""Test the emotion-tree pipeline idea: does its hierarchy come from structure or margins?

The proposal is to replace the linear sum with a hierarchy built from an emotion-word
probe. Replicating the described design:

  1. for each item, append a probe phrase and read next-token probabilities over an
     emotion vocabulary, giving Y (N_items x V_emotions)
  2. matching matrix C = Y^T Y, which measures how often two emotions are produced in
     similar contexts
  3. orient the hierarchy by C_ab / sum_i C_ai > t and C_ab / sum_i C_ib <
     C_ab / sum_i C_ai, i.e. a is the parent of b

Two things are checked.

First, a mathematical observation that the second condition simplifies. Since C_ab > 0
it cancels, leaving sum_i C_ai < sum_i C_ib. So the orientation rule uses only the row
and column sums of C, not the joint entries. If that is right, the hierarchy is a
function of the marginals alone, and it will appear on any matrix with uneven margins
whether or not a hierarchy exists.

Second, an empirical check of the properties claimed for the approach: order
invariance, and whether the resulting graph carries information that separates corpora.

Run with a small model if the 7B is too slow:
    python exploration/emotion_tree_probe.py --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
from pacsp_core import load_samples  # noqa: E402

DATA = ROOT / "data"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")

# A compact Chinese emotion lexicon. The source paper uses 135 words; the count is not
# critical to what is being tested here, and a smaller set keeps the forward passes cheap.
EMOTIONS = [
    "喜悦", "快乐", "高兴", "兴奋", "欣喜", "愉快", "满足", "幸福", "欣慰", "感动",
    "乐观", "希望", "期待", "憧憬", "向往", "自信", "骄傲", "自豪", "得意", "轻松",
    "平静", "安宁", "宁静", "放松", "释然", "坦然", "安心", "舒畅", "惬意", "从容",
    "爱", "喜爱", "热爱", "眷恋", "思念", "怀念", "牵挂", "温柔", "亲切", "亲密",
    "悲伤", "难过", "伤心", "痛苦", "悲痛", "哀伤", "凄凉", "忧郁", "沮丧", "失落",
    "孤独", "寂寞", "空虚", "惆怅", "迷茫", "困惑", "无奈", "无力", "绝望", "消沉",
    "愤怒", "生气", "恼怒", "不满", "怨恨", "厌恶", "反感", "嫉妒", "轻蔑", "敌意",
    "恐惧", "害怕", "惊恐", "焦虑", "紧张", "担忧", "不安", "惶恐", "畏惧", "战栗",
    "惊讶", "震惊", "意外", "诧异", "疑惑", "好奇", "兴趣", "关注", "专注", "投入",
    "羞愧", "内疚", "懊悔", "自责", "尴尬", "羞耻", "负罪", "后悔", "遗憾", "惋惜",
    "感激", "感谢", "敬佩", "尊敬", "崇拜", "仰慕", "信任", "依赖", "认同", "归属",
    "思念2", "平静2", "讽刺", "嘲弄", "冷淡", "漠然", "麻木", "厌倦", "疲惫", "倦怠",
    "鼓励", "激励", "振奋", "鼓舞", "坚定", "决心", "勇气", "坚韧", "执着", "毅力",
]
# drop the two placeholder entries that only pad the list
EMOTIONS = [e for e in EMOTIONS if not e.endswith("2")]
EMOTIONS = list(dict.fromkeys(EMOTIONS))

PROBE = "这段素材的核心情绪是"


def load_model(name, device=None, four_bit=True):
    """Load the model on the GPU, 4-bit by default.

    bfloat16 weights need about 14 GB against roughly 10.9 GB of free VRAM, so the
    model cannot be placed whole. nf4 4-bit quantisation brings it to about 5.2 GB,
    which fits with room for activations and runs a forward pass in roughly 56 ms
    against 5523 ms on the CPU, a factor of about 100.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    if device == "cuda" and four_bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
        kwargs["dtype"] = torch.float16
    else:
        kwargs["dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
        if device == "cuda":
            kwargs["device_map"] = "auto"

    try:
        mdl = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.float16
        mdl = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    mdl.eval()
    if device == "cuda":
        print(f"  cuda: {torch.cuda.get_device_name(0)}  "
              f"VRAM {torch.cuda.memory_allocated()/2**30:.2f} GB")
    return tok, mdl, torch


def emotion_token_ids(tok, words):
    """Map each emotion word to a single token id where possible."""
    ids = {}
    for w in words:
        for variant in (w, " " + w):
            enc = tok.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                ids[w] = enc[0]
                break
    return ids


def probe_matrix(texts, tok, mdl, torch, ids, max_len=256, device=None):
    """Y[i, j] = probability the model puts on emotion j after item i + probe."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    words = list(ids.keys())
    cols = np.zeros((len(texts), len(words)), dtype=np.float64)
    ids_list = [ids[w] for w in words]
    for i, t in enumerate(texts):
        prompt = (t.strip().replace("\n", " ")[:400] + "\n" + PROBE)
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = mdl(**enc)
        logits = out.logits[0, -1, :].float()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        cols[i] = probs[ids_list]
        if (i + 1) % 5 == 0:
            print(f"    probed {i+1}/{len(texts)}")
    total = cols.sum(axis=1, keepdims=True)
    return cols, words, total


def build_hierarchy(C, t=0.02):
    """Apply the described rule. Returns directed edges (parent, child, value)."""
    V = C.shape[0]
    rs = C.sum(axis=1)
    cs = C.sum(axis=0)
    edges = []
    for a in range(V):
        for b in range(V):
            if a == b or rs[a] <= 0 or cs[b] <= 0:
                continue
            cond1 = C[a, b] / rs[a] > t
            cond2 = C[a, b] / cs[b] < C[a, b] / rs[a]
            if cond1 and cond2:
                edges.append((a, b, float(C[a, b])))
    return edges, rs, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms", nargs="*", default=["poem", "machine_poem"])
    ap.add_argument("--limit", type=int, default=31)
    args = ap.parse_args()

    print(f"  loading {args.model}")
    tok, mdl, torch = load_model(args.model)
    ids = emotion_token_ids(tok, EMOTIONS)
    print(f"  emotion words mapped to single tokens: {len(ids)}/{len(EMOTIONS)}")
    if len(ids) < 20:
        print("  too few single-token emotions; aborting")
        return 1

    report = {}
    for arm in args.arms:
        p = DATA / arm
        if not p.is_dir():
            print(f"  [skip] {arm}")
            continue
        texts, _ = load_samples(p)
        texts = texts[:args.limit]
        print(f"\n=== {arm}  n={len(texts)} ===")
        Y, words, _ = probe_matrix(texts, tok, mdl, torch, ids)
        C = Y.T @ Y
        print(f"    Y {Y.shape}  C {C.shape}  "
              f"Y row sums: {Y.sum(axis=1).min():.4f}-{Y.sum(axis=1).max():.4f}")

        edges, rs, cs = build_hierarchy(C)
        print(f"    hierarchy edges: {len(edges)} of {len(words)*(len(words)-1)} possible")
        top = sorted(edges, key=lambda e: -e[2])[:8]
        print(f"    strongest edges (parent -> child, weight):")
        for a, b, w in top:
            print(f"      {words[a]} -> {words[b]}   {w:.3e}")

        # the key question: is the rule driven by margins alone?
        # recompute with the joint entries replaced by a rank-1 outer product, which
        # has the same margins up to scale but no interaction structure at all
        r = rs / rs.sum()
        c = cs / cs.sum()
        C_rank1 = np.outer(r, c) * C.sum()
        e2, _, _ = build_hierarchy(C_rank1)
        same = len(set((a, b) for a, b, _ in edges) & set((a, b) for a, b, _ in e2))
        print(f"    rank-1 reconstruction (same margins, no structure): "
              f"{len(e2)} edges, {same} identical to the real graph")

        # order invariance: C = Y^T Y is a set-level statistic, so permuting items
        # must leave it unchanged. Verified rather than assumed.
        perm = np.random.default_rng(1).permutation(len(texts))
        C_perm = (Y[perm].T @ Y[perm])
        print(f"    C unchanged under item permutation: "
              f"{np.allclose(C, C_perm, atol=1e-12)}  "
              f"(max abs diff {np.abs(C-C_perm).max():.2e})")

        report[arm] = {
            "n": len(texts), "vocab": len(words), "edges": len(edges),
            "rank1_edges": len(e2), "rank1_identical": int(same),
            "C_permutation_invariant": bool(np.allclose(C, C_perm, atol=1e-12)),
            "top_edges": [{"parent": words[a], "child": words[b], "w": w}
                          for a, b, w in top],
            "Y_row_sum_min": float(Y.sum(axis=1).min()),
            "Y_row_sum_max": float(Y.sum(axis=1).max()),
        }

    f = ROOT / "records_centroid" / f"emotion_tree_{STAMP}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
