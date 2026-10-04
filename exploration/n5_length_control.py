"""N5: are the five pairs length-matched, and does length drive D?

Section 4.3 reports five same-domain human-versus-machine pairs. The reviewer's point is
that same-domain is not the same as same-length, and if length drives D then the whole
result is a length effect in disguise.

This is the check that could overturn section 4.3, so it comes first.

Two parts.

  audit    the byte and character distribution of each arm in each pair, and whether the
           two arms of a pair are matched
  control  D recomputed after trimming every text to a common length, so the comparison
           is length-matched by construction rather than by luck. A second variant trims
           to the shortest arm's median, which is the strictest available without
           discarding corpus content.
"""

import json
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
PAIRS = [("poem", "machine_poem"), ("lyrics", "machine_lyrics"),
         ("techdoc", "machine_techdoc"),
         ("hc3_human_medicine", "hc3_ai_medicine"),
         ("hc3_human_openqa", "hc3_ai_openqa")]


def sq(A, B):
    aa = np.sum(A * A, axis=1)[:, None]
    bb = np.sum(B * B, axis=1)[None, :]
    return np.maximum(aa + bb - 2.0 * (A @ B.T), 0.0)


def mpd(E):
    n = len(E)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(sq(E, E)[iu]).mean())


def trim(text, target_chars):
    """Cut to whole sentences at or below the target; fall back to a hard cut."""
    t = " ".join(text.split())
    if len(t) <= target_chars:
        return t
    import re
    parts = [p for p in re.split(r"(?<=[。！？；!?;])", t) if p.strip()]
    acc = ""
    for p in parts:
        if len(acc) + len(p) > target_chars:
            break
        acc += p
    if len(acc) < target_chars * 0.4:
        acc = t[:target_chars]
    return acc


def load(arm):
    texts, _ = pacsp_core.load_samples(DATA / arm)
    return texts


def main():
    out = {}

    # ------------------------------------------------------------------ audit
    print("=== length audit: are the pairs matched? ===")
    print(f"  {'pair':<38} {'human bytes':>12} {'machine':>10} {'ratio':>7} "
          f"{'human chars':>12} {'machine':>10} {'ratio':>7}")
    for h, m in PAIRS:
        if not (DATA / h).is_dir() and not DATA.joinpath(h).exists():
            pass
        if not (DATA / h).is_dir() or not (DATA / m).is_dir():
            print(f"  {h} / {m}: missing")
            continue
        th, tm = load(h), load(m)
        bh = np.array([len(t.encode("utf-8")) for t in th])
        bm = np.array([len(t.encode("utf-8")) for t in tm])
        ch = np.array([len(t) for t in th])
        cm = np.array([len(t) for t in tm])
        label = f"{h.split('_')[-1]} ({len(th)} vs {len(tm)})"
        print(f"  {label:<38} {bh.mean():>12.0f} {bm.mean():>10.0f} "
              f"{bh.mean()/bm.mean():>7.2f} {ch.mean():>12.0f} {cm.mean():>10.0f} "
              f"{ch.mean()/cm.mean():>7.2f}")
        out.setdefault("audit", {})[h.split("_")[-1]] = {
            "human_bytes_mean": int(bh.mean()), "machine_bytes_mean": int(bm.mean()),
            "human_chars_mean": int(ch.mean()), "machine_chars_mean": int(cm.mean()),
            "byte_ratio": round(float(bh.mean() / bm.mean()), 4),
            "char_ratio": round(float(ch.mean() / cm.mean()), 4),
        }

    # ------------------------------------------------------------------ control
    print()
    print("=== length-matched control: trim every text to a common character budget ===")
    print("  target = the smaller arm's median character length, so nothing is invented")
    print(f"  {'pair':<16} {'target':>7} {'orig ratio':>11} {'trimmed ratio':>14} "
          f"{'change':>8}")
    for h, m in PAIRS:
        if not (DATA / h).is_dir() or not (DATA / m).is_dir():
            continue
        th, tm = load(h), load(m)
        med_h = int(np.median([len(t) for t in th]))
        med_m = int(np.median([len(t) for t in tm]))
        target = min(med_h, med_m)
        if target < 40:
            target = max(40, target)

        Eh = pacsp_core.compute_embeddings([trim(t, target) for t in th],
                                           model_name=MODEL)
        Em = pacsp_core.compute_embeddings([trim(t, target) for t in tm],
                                           model_name=MODEL)
        orig = mpd(pacsp_core.compute_embeddings(th, model_name=MODEL)) / \
            mpd(pacsp_core.compute_embeddings(tm, model_name=MODEL))
        trimmed = mpd(Eh) / mpd(Em)
        key = h.split("_")[-1]
        print(f"  {key:<16} {target:>7} {orig:>11.4f} {trimmed:>14.4f} "
              f"{trimmed/orig:>8.3f}")
        out.setdefault("control", {})[key] = {
            "target_chars": target,
            "orig_ratio": round(float(orig), 4),
            "trimmed_ratio": round(float(trimmed), 4),
            "human_bytes_after": int(np.mean([len(t.encode()) for t in
                                              [trim(x, target) for x in th]])),
            "machine_bytes_after": int(np.mean([len(t.encode()) for t in
                                                [trim(x, target) for x in tm]])),
        }

    # ------------------------------------------------------------------ verdict
    print()
    print("=== verdict ===")
    ctrl = out.get("control", {})
    if ctrl:
        orig = [v["orig_ratio"] for v in ctrl.values()]
        trim_ = [v["trimmed_ratio"] for v in ctrl.values()]
        below_o = sum(1 for v in orig if v < 1)
        below_t = sum(1 for v in trim_ if v < 1)
        print(f"  direction below 1:   original {below_o}/{len(orig)}   "
              f"trimmed {below_t}/{len(trim_)}")
        print(f"  spread:              original {max(orig)/min(orig):.2f}x   "
              f"trimmed {max(trim_)/min(trim_):.2f}x")
        # did any pair change direction?
        flips = [k for k, v in ctrl.items()
                 if (v["orig_ratio"] < 1) != (v["trimmed_ratio"] < 1)]
        print(f"  pairs whose direction flipped after trimming: {flips or 'none'}")
        shifts = [abs(v["trimmed_ratio"] / v["orig_ratio"] - 1) for v in ctrl.values()]
        print(f"  largest relative shift in the ratio: {max(shifts):.1%}")
        out["verdict"] = {
            "below_1_original": below_o, "below_1_trimmed": below_t,
            "spread_original": round(max(orig) / min(orig), 4),
            "spread_trimmed": round(max(trim_) / min(trim_), 4),
            "direction_flips": flips,
            "max_shift": round(max(shifts), 4),
        }

    f = M / "records_centroid" / "n5_length_control.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
