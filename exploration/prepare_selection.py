"""Prepare the selection task: present two drafts per prompt, length-matched.

The reader's task is to pick the better essay. If one draft is systematically longer
the choice collapses into a length preference, which is not the variable under study.
The r1 drafts average about 20 percent shorter than the qwen2.5 drafts, so both are
trimmed to whole sentences at a common target before being shown, and the target is
the shorter of the two so nothing is fabricated.

Output for each prompt: a plain-text comparison sheet in selection/ with the two
candidates as "候选一" and "候选二", their order randomised per prompt by a seed
derived from the index so the position carries no information, and a machine-readable
key recording which model produced which slot.
"""

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
A = ROOT / "gaokao_ai_raw"          # qwen2.5:7b
B = ROOT / "gaokao_ai_raw_b"        # deepseek-r1:14b
PROMPTS = ROOT / "gaokao_prompts"
OUT = ROOT / "gaokao_selection"

SPLIT = re.compile(r"(?<=[。！？；])")
TARGET = 800          # characters, close to both models' natural length
FLOOR = 600


def trim(text, target):
    """Cut at sentence boundaries to at most `target` characters."""
    text = " ".join(text.split())
    parts = [p for p in SPLIT.split(text) if p.strip()]
    acc = ""
    for p in parts:
        if len(acc) + len(p) > target and len(acc) >= FLOOR:
            break
        acc += p
    return acc if len(acc) >= FLOOR else text


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()

    files = sorted(PROMPTS.glob("[0-9][0-9][0-9].txt"))
    key = []
    ready = 0
    for f in files:
        idx = int(f.stem)
        pa, pb = A / f.name, B / f.name
        if not (pa.exists() and pb.exists()):
            continue
        ta, tb = pa.read_text(encoding="utf-8"), pb.read_text(encoding="utf-8")
        # trim both to the same ceiling
        ta2, tb2 = trim(ta, TARGET), trim(tb, TARGET)

        rng = random.Random(idx * 7919)
        flip = rng.random() < 0.5
        first, second = (tb2, ta2) if flip else (ta2, tb2)
        first_model, second_model = (("r1", "qwen") if flip else ("qwen", "r1"))

        prompt = f.read_text(encoding="utf-8").strip()
        sheet = (
            f"题目 {idx:03d}\n"
            f"{'=' * 70}\n"
            f"【题面】\n{prompt}\n\n"
            f"{'=' * 70}\n"
            f"【候选一】（{len(first)} 字符）\n{first}\n\n"
            f"{'=' * 70}\n"
            f"【候选二】（{len(second)} 字符）\n{second}\n"
        )
        (OUT / f"{idx:03d}.txt").write_text(sheet, encoding="utf-8", newline="\n")
        key.append({
            "index": idx,
            "candidate_1": first_model,
            "candidate_2": second_model,
            "chars_1": len(first),
            "chars_2": len(second),
            "original_chars": {"qwen": len(ta.strip()), "r1": len(tb.strip())},
        })
        ready += 1

    (OUT / "_key.json").write_text(json.dumps({
        "purpose": ("which draft the reader picks; the key maps candidate slots to "
                    "models and must not be read before choosing"),
        "target_chars": TARGET,
        "floor_chars": FLOOR,
        "count": ready,
        "items": key,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  comparison sheets: {ready}")
    if key:
        c1 = [k["chars_1"] for k in key]
        c2 = [k["chars_2"] for k in key]
        print(f"  candidate 1 chars: min {min(c1)} max {max(c1)} mean {sum(c1)//len(c1)}")
        print(f"  candidate 2 chars: min {min(c2)} max {max(c2)} mean {sum(c2)//len(c2)}")
        flips = sum(1 for k in key if k["candidate_1"] == "r1")
        print(f"  r1 placed first in {flips}/{ready} sheets (randomised)")
    print(f"  written to {OUT}")
    print(f"  selection answer file expected at {OUT / '_choice.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
