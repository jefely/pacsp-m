"""Separate the selection effect from the model-preference effect.

The chosen/unchosen contrast gave C_T 3.5483 against 3.8648, a ratio of 0.92. But the
reader picked r1 in 9 of 13 decisions, so "chosen" is largely a proxy for "r1". If r1
drafts simply score lower than qwen drafts, the measured difference is a model
difference wearing the label of selection.

This computes C_T for each model's full 31-draft arm, so the two effects can be told
apart. It also reports the chosen/unchosen contrast restricted to decisions that went
each way, which is the within-model test.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = ROOT / "records_centroid"
DATA = ROOT / "data"


def run(arms):
    # The six-layer build lives in PACSP-ID, not here; this script only analyses the
    # arms it produced. Fail loudly rather than silently doing nothing.
    builder = ROOT.parent / "PACSP-ID" / "scripts" / "pacsp_build.py"
    if not builder.is_file():
        print(f"  this step needs {builder}, which is not present")
        return 1
    cmd = [PY, "-X", "utf8", str(builder),
           "--innov", "--target", "centroid", "--datasets", *arms,
           "--out", str(OUT)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    got = {}
    current = None
    for line in (r.stdout or "").splitlines():
        m = re.search(r"C_T = ([\d.]+) Se", line)
        if m:
            current = float(m.group(1))
        m2 = re.search(r"DONE:\s*(\S+)", line)
        if m2 and current is not None:
            got[m2.group(1)] = current
            current = None
    return got, r.returncode


def main():
    print("=== full arms, both models, all 31 prompts ===")
    full, rc = run(["gaokao_ai_raw", "gaokao_ai_raw_b"])
    q, r1 = full.get("gaokao_ai_raw"), full.get("gaokao_ai_raw_b")
    print(f"  gaokao_ai_raw   (qwen2.5:7b)      C_T = {q}")
    print(f"  gaokao_ai_raw_b (deepseek-r1:14b) C_T = {r1}")
    if q and r1:
        print(f"  r1 / qwen ratio = {r1/q:.3f}")
        print("  -> a ratio well below 1 means r1 drafts score lower on their own,")
        print("     so a chosen/unchosen gap can be produced by model preference alone")

    print()
    print("=== the reader's 13 decisions ===")
    sel = DATA / "gaokao_selection"
    key = json.loads((sel / "_key.json").read_text(encoding="utf-8"))
    picks = {}
    for line in (sel / "_choice.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            picks[int(parts[0])] = parts[1].upper()

    by_model = {}
    for it in key["items"]:
        i = it["index"]
        if i not in picks:
            continue
        slot = "candidate_1" if picks[i] == "A" else "candidate_2"
        m = it[slot]
        by_model[m] = by_model.get(m, 0) + 1
    n = sum(by_model.values())
    print(f"  chosen model counts: {by_model}  (n={n})")
    if by_model:
        top = max(by_model, key=by_model.get)
        print(f"  most chosen: {top} in {by_model[top]}/{n} = "
              f"{by_model[top]/n:.0%}")

    print()
    print("=== within-model test: do r1 choices differ from qwen choices? ===")
    # for each decision, was the chosen draft r1 or qwen, and did the reader pick A?
    rows = []
    for it in key["items"]:
        i = it["index"]
        if i not in picks:
            continue
        slot = "candidate_1" if picks[i] == "A" else "candidate_2"
        chosen_model = it[slot]
        other_model = it["candidate_2"] if picks[i] == "A" else it["candidate_1"]
        rows.append({"index": i, "pick": picks[i],
                     "chosen": chosen_model, "unchosen": other_model,
                     "chars_chosen": it["chars_1"] if picks[i] == "A" else it["chars_2"],
                     "chars_unchosen": it["chars_2"] if picks[i] == "A" else it["chars_1"]})
    both = [r for r in rows if r["chosen"] != r["unchosen"]]
    same = [r for r in rows if r["chosen"] == r["unchosen"]]
    print(f"  decisions between different models: {len(both)}")
    print(f"  decisions between the same model  : {len(same)}")
    for r in both:
        print(f"    {r['index']:03d}  {r['pick']}  chose {r['chosen']:<5} "
              f"({r['chars_chosen']} ch) over {r['unchosen']:<5} "
              f"({r['chars_unchosen']} ch)")

    print()
    print("=== length control check ===")
    cc = [r["chars_chosen"] for r in rows]
    cu = [r["chars_unchosen"] for r in rows]
    if cc:
        print(f"  chosen   mean {sum(cc)//len(cc)} chars")
        print(f"  unchosen mean {sum(cu)//len(cu)} chars")
        print(f"  ratio {sum(cu)/sum(cc):.3f}  "
              f"({'balanced' if abs(sum(cu)/sum(cc)-1) < 0.05 else 'NOT balanced'})")

    (OUT / "selection_analysis.json").write_text(json.dumps({
        "full_arm_ct": {"qwen": q, "r1": r1},
        "chosen_model_counts": by_model,
        "decisions": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  written {OUT / 'selection_analysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
