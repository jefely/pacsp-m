"""N6c: harden the H1 result. Discard non-essays, then control for the remaining length gap.

N6b found that the techdoc reversal disappears once the machine arm is generated on the
same topics: the ratio goes from 1.4426 to 0.9205. Two things could still undermine that.

First, the generation output is not uniformly an essay. Several texts open with 好的。 or
能。 or 这是一个极具挑战性的请求, which are conversational replies rather than technical
documents. If those are included the machine arm's spread is inflated by genre mixing
rather than by content, which would make 0.9205 too generous to the hypothesis.

Second, the topic-matched arm is still 4.42x shorter than the human arm. N5 showed length
does not drive D across the original pairs, but that was tested on the original arms, so
the check is repeated here on the new pair.

Both are done, and the reversal claim is then restated only as far as the data supports.
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(M))
import pacsp_core  # noqa: E402

DATA = M / "data"
MODEL = "BAAI/bge-large-zh-v1.5"

# openers that mark a conversational reply rather than a document. The first version of
# this pattern ended the group immediately after 好的 or 能, so it never matched 好的。 and
# discarded nothing. The group is now followed by optional punctuation.
CHATTY = re.compile(
    r"^\s*(好的|能|可以|当然|完全可以|这是一个|这是一个极其|这是一个非常|"
    r"这是一个极具|以下是)[，。！？、,.\s]")


def is_essay(t):
    """A generated document, judged by opener and length."""
    s = " ".join(t.split())
    if len(s) < 400:
        return False, "short"
    if CHATTY.match(s):
        return False, "chatty opener"
    # a body that is mostly one short line is also a reply, not a document
    if len(s.splitlines()) <= 2 and len(s) < 800:
        return False, "too short to be a document"
    return True, "kept"


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(sq(E, E)[iu]).mean())


def trim(text, target):
    t = " ".join(text.split())
    if len(t) <= target:
        return t
    parts = [p for p in re.split(r"(?<=[。！？；!?;])", t) if p.strip()]
    acc = ""
    for p in parts:
        if len(acc) + len(p) > target:
            break
        acc += p
    return acc if len(acc) >= target * 0.4 else t[:target]


def main():
    human, _ = pacsp_core.load_samples(DATA / "techdoc")
    gen, files = pacsp_core.load_samples(DATA / "machine_techdoc2")
    out = {}

    # ------------------------------------------------------------------ filter
    print("=== discarding conversational replies from the generated arm ===")
    keep, drop = [], []
    for f, t in zip(files, gen):
        ok, why = is_essay(t)
        if ok:
            keep.append(t)
        else:
            drop.append((f.name, " ".join(t.split())[:46], why))
    print(f"  kept {len(keep)}, dropped {len(drop)}")
    for name, head, why in drop:
        print(f"    {name}: {why:<24} {head}")
    out["filter"] = {"kept": len(keep), "dropped": len(drop),
                     "dropped_detail": [{"file": n, "head": h, "reason": w}
                                        for n, h, w in drop]}

    # ------------------------------------------------------------------ measure
    print("\n=== D after filtering ===")
    E_h = pacsp_core.compute_embeddings(human, model_name=MODEL)
    E_g_all = pacsp_core.compute_embeddings(gen, model_name=MODEL)
    E_g = pacsp_core.compute_embeddings(keep, model_name=MODEL)
    d_h, d_ga, d_g = mpd(E_h), mpd(E_g_all), mpd(E_g)
    print(f"  techdoc (human)              n={len(human):>3}  D={d_h:.4f}")
    print(f"  machine_techdoc2 (all)       n={len(gen):>3}  D={d_ga:.4f}")
    print(f"  machine_techdoc2 (filtered)  n={len(keep):>3}  D={d_g:.4f}")
    print(f"  ratio all      : {d_h/d_ga:.4f}")
    print(f"  ratio filtered : {d_h/d_g:.4f}")
    out["filtered"] = {"D_human": round(d_h, 4), "D_all": round(d_ga, 4),
                       "D_filtered": round(d_g, 4),
                       "ratio_all": round(d_h / d_ga, 4),
                       "ratio_filtered": round(d_h / d_g, 4)}

    # ------------------------------------------------------------------ length
    print("\n=== length control on the topic-matched pair ===")
    med_h = int(np.median([len(t) for t in human]))
    med_g = int(np.median([len(t) for t in keep]))
    target = min(med_h, med_g)
    print(f"  median chars: human {med_h}, machine {med_g}, target {target}")
    E_h_t = pacsp_core.compute_embeddings([trim(t, target) for t in human],
                                          model_name=MODEL)
    E_g_t = pacsp_core.compute_embeddings([trim(t, target) for t in keep],
                                          model_name=MODEL)
    d_h_t, d_g_t = mpd(E_h_t), mpd(E_g_t)
    print(f"  trimmed: human {d_h_t:.4f}  machine {d_g_t:.4f}  "
          f"ratio {d_h_t/d_g_t:.4f}")
    print(f"  change vs untrimmed filtered ratio: "
          f"{(d_h_t/d_g_t)/(d_h/d_g):.3f}")
    out["length_control"] = {"target": target, "D_human_trimmed": round(d_h_t, 4),
                             "D_machine_trimmed": round(d_g_t, 4),
                             "ratio_trimmed": round(d_h_t / d_g_t, 4)}

    # ------------------------------------------------------------------ verdict
    print("\n=== verdict ===")
    r = d_h_t / d_g_t
    survives = r > 1
    print(f"  topic-matched, filtered, length-controlled ratio: {r:.4f}")
    print(f"  reversal survives: {survives}")
    out["verdict"] = {
        "final_ratio": round(r, 4),
        "reversal_survives": bool(survives),
        "original_ratio": 1.4426,
        "interpretation": ("reversal gone" if not survives
                           else "reversal survives the stricter test"),
    }
    if not survives:
        print("  -> the section 4.5 reversal was an artefact of comparing different")
        print("     topics; on matched topics the human arm has the lower D, as in")
        print("     the other four domains")

    f = M / "records_centroid" / "n6c_h1_hardened.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
