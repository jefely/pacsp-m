"""N6b: H1 test. Generate a machine arm on the human techdoc topics and re-measure.

The audit found the existing pair badly confounded. The human arm averages 19,545 bytes
on topics about self-attention credit and digital currency; the machine arm averages 2,350
bytes on Python virtual environments. That is an 8.3x length gap and a complete topic
mismatch, so the reversal in section 4.5 may be an artefact of the mismatch rather than
the genre effect H1 proposes.

H1 says human technical writing is itself more dispersed while AI technical writing is
more templated. Testing it needs a machine arm on the same topics. This generates one:
each human techdoc is summarised into a topic prompt by its own opening, a single-shot
technical document is generated per topic, and D is recomputed.

Three arms are then compared:
    techdoc            the original human arm
    machine_techdoc    the original machine arm, different topics, 8.3x shorter
    machine_techdoc2   the new machine arm, same topics, generated once each

If the reversal disappears when topics match, the original was a topic artefact. If it
persists, H1 survives this test.
"""

import json
import os
import re
import socket
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
OUT = DATA / "machine_techdoc2"
MODEL_EMB = "BAAI/bge-large-zh-v1.5"
OLLAMA = "http://127.0.0.1:11434"
GEN_MODEL = "qwen2.5:7b"
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
socket.setdefaulttimeout(600)

PROMPT = ("请就以下主题撰写一篇技术说明文，不少于 800 字。\n\n"
          "主题：{topic}\n\n"
          "要求：结构完整，术语准确，论述具体。直接输出正文，不要输出解释或标题说明。")


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(sq(E, E)[iu]).mean())


def topic_of(text):
    """First sentence, capped, as the topic."""
    t = " ".join(text.split())
    m = re.split(r"(?<=[。！？；])", t)
    return (m[0] if m else t)[:120]


def generate(prompt, seed):
    payload = json.dumps({
        "model": GEN_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.8, "top_p": 0.9, "num_predict": 2000,
                    "seed": seed}}).encode()
    req = urllib.request.Request(OLLAMA + "/api/generate", data=payload,
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()).get("response", "").strip()


def main():
    human, _ = pacsp_core.load_samples(DATA / "techdoc")
    print(f"  human techdoc: {len(human)} texts, "
          f"mean {int(np.mean([len(t) for t in human]))} chars")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n=== generating {len(human)} machine texts on the human topics ===")
    made = 0
    for i, t in enumerate(human, 1):
        dest = OUT / f"{i:03d}.txt"
        if dest.exists() and dest.stat().st_size > 200:
            made += 1
            continue
        topic = topic_of(t)
        t0 = time.time()
        try:
            body = generate(PROMPT.format(topic=topic), seed=7000 + i)
        except Exception as e:
            print(f"  [{i:>2}] FAILED {type(e).__name__}: {str(e)[:80]}")
            continue
        dest.write_text(body + "\n", encoding="utf-8", newline="\n")
        made += 1
        print(f"  [{i:>2}] {len(body):>5} chars in {time.time()-t0:>4.1f}s  "
              f"{topic[:52]}")
    print(f"  generated {made}/{len(human)}")

    # ------------------------------------------------------------------ measure
    print("\n=== D on the three arms ===")
    arms = ["techdoc", "machine_techdoc", "machine_techdoc2"]
    Embs = {}
    stats = {}
    for arm in arms:
        p = DATA / arm
        if not p.is_dir():
            print(f"  [skip] {arm}")
            continue
        texts, _ = pacsp_core.load_samples(p)
        Embs[arm] = pacsp_core.compute_embeddings(texts, model_name=MODEL_EMB)
        d = mpd(Embs[arm])
        stats[arm] = {
            "n": len(texts), "D": round(d, 4),
            "chars_mean": int(np.mean([len(t) for t in texts])),
            "bytes_mean": int(np.mean([len(t.encode()) for t in texts])),
        }
        print(f"  {arm:<20} n={len(texts):>3}  D={d:.4f}  "
              f"mean {stats[arm]['chars_mean']:>6} chars")

    # ------------------------------------------------------------------ verdict
    print("\n=== H1 test ===")
    out = {"stats": stats, "stamp": STAMP}
    if "techdoc" in stats and "machine_techdoc2" in stats:
        r_old = stats["techdoc"]["D"] / stats["machine_techdoc"]["D"]
        r_new = stats["techdoc"]["D"] / stats["machine_techdoc2"]["D"]
        print(f"  original pair (different topics):  "
              f"{stats['techdoc']['D']:.4f} / {stats['machine_techdoc']['D']:.4f} "
              f"= {r_old:.4f}")
        print(f"  topic-matched pair (new):          "
              f"{stats['techdoc']['D']:.4f} / {stats['machine_techdoc2']['D']:.4f} "
              f"= {r_new:.4f}")
        print()
        print(f"  length gap, original : "
              f"{stats['techdoc']['bytes_mean']/stats['machine_techdoc']['bytes_mean']:.2f}x")
        print(f"  length gap, matched  : "
              f"{stats['techdoc']['bytes_mean']/stats['machine_techdoc2']['bytes_mean']:.2f}x")
        print()
        survives = r_new > 1
        print(f"  reversal survives topic matching: {survives}")
        if survives:
            print("  -> H1 (genre effect) is supported: human technical writing is")
            print("     more dispersed even against machine text on the same topics")
        else:
            print("  -> H1 is NOT supported by this test: the original reversal was")
            print("     an artefact of comparing different topics")
        out["verdict"] = {
            "ratio_original": round(r_old, 4),
            "ratio_topic_matched": round(r_new, 4),
            "reversal_survives": bool(survives),
            "length_gap_original": round(
                stats["techdoc"]["bytes_mean"] /
                stats["machine_techdoc"]["bytes_mean"], 4),
            "length_gap_matched": round(
                stats["techdoc"]["bytes_mean"] /
                stats["machine_techdoc2"]["bytes_mean"], 4),
        }

    f = M / "records_centroid" / f"n6b_h1_{STAMP}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
