"""Build a length-matched two-arm corpus from HC3-Chinese.

What HC3-Chinese gives: for each question, human answers and ChatGPT answers. Two
arms, prompt-parallel, same language and era -- a genuine authorship control. What it
does NOT give: the paper's middle arm, where a human curates AI output.

Length matching is mandatory here, not cosmetic. In the finance domain human answers
average roughly 150 characters against ChatGPT's 450, and C_T sums unnormalised
embedding distances, so an unmatched comparison would largely measure length. Each
answer is therefore trimmed to whole sentences at a common target length, and any
item that cannot reach the floor is dropped.

Output layout matches the existing arms: data/<arm>/001.txt ... 031.txt.
"""

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "dsh-corpus-build"}
DS = "Hello-SimpleAI/HC3-Chinese"
ROOT = Path(__file__).resolve().parent.parent / "data"

# sentence-final punctuation in Chinese, kept with the sentence
SPLIT = re.compile(r"(?<=[。！？；!?;])")


def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def fetch(cfg, offset, length):
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           + urllib.parse.quote(DS)
           + f"&config={urllib.parse.quote(cfg)}&split=train"
           + f"&offset={offset}&length={length}")
    return json.loads(get(url))


def trim_to(text, target, floor):
    """Cut at a sentence boundary near `target`; return None if below `floor`."""
    text = " ".join(str(text).split())
    if len(text) < floor:
        return None
    parts = [p for p in SPLIT.split(text) if p.strip()]
    if not parts:
        return None
    best = None
    acc = ""
    for p in parts:
        acc += p
        if len(acc) < floor:
            continue
        if best is None:
            best = acc
        # prefer the chunk whose length is closest to target
        if abs(len(acc) - target) <= abs(len(best) - target):
            best = acc
        if len(acc) >= target:
            break
    return best if best and len(best) >= floor else None


def build(cfg, n, target, floor, out_human, out_ai):
    got = []
    offset = 0
    while len(got) < n and offset < 4000:
        try:
            page = fetch(cfg, offset, 100)
        except Exception as e:
            print(f"    fetch offset={offset} failed: {type(e).__name__}")
            break
        rows = page.get("rows") or []
        if not rows:
            break
        for rw in rows:
            row = rw["row"]
            hum = row.get("human_answers") or []
            gpt = row.get("chatgpt_answers") or []
            if not hum or not gpt:
                continue
            h = trim_to(max(hum, key=len), target, floor)
            a = trim_to(max(gpt, key=len), target, floor)
            if h and a:
                got.append((row.get("question", ""), h, a))
                if len(got) >= n:
                    break
        offset += len(rows)

    (ROOT / out_human).mkdir(parents=True, exist_ok=True)
    (ROOT / out_ai).mkdir(parents=True, exist_ok=True)
    for d in (ROOT / out_human, ROOT / out_ai):
        for f in d.glob("*.txt"):
            f.unlink()

    meta = []
    for i, (q, h, a) in enumerate(got, 1):
        (ROOT / out_human / f"{i:03d}.txt").write_text(h + "\n", encoding="utf-8",
                                                        newline="\n")
        (ROOT / out_ai / f"{i:03d}.txt").write_text(a + "\n", encoding="utf-8",
                                                    newline="\n")
        meta.append({"src_index": i, "question": q,
                     "human_chars": len(h), "ai_chars": len(a)})

    prov = {
        "source_dataset": DS,
        "config": cfg,
        "license": "cc-by-sa-4.0",
        "arms": [out_human, out_ai],
        "what_this_is": (
            "Prompt-parallel authorship control. Each question carries one human "
            "answer and one ChatGPT answer, so topic and prompt are held constant "
            "while authorship varies. This is the strongest publicly available "
            "substitute for a same-source three-arm design."
        ),
        "what_this_is_not": (
            "It has no middle arm. The paper's axis is whether human selection and "
            "curation was applied, and HC3's ChatGPT answers are raw generations "
            "with no curation step, while the human answers involve no model. So "
            "this contrasts human authorship against machine authorship, not "
            "curated against uncurated machine output."
        ),
        "length_matching": {
            "method": "each answer cut to whole sentences near a common target",
            "target_chars": target,
            "floor_chars": floor,
        },
        "count": len(meta),
        "items": meta,
    }
    (ROOT / out_human / "_PROVENANCE.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=2), encoding="utf-8")

    hs = [m["human_chars"] for m in meta]
    as_ = [m["ai_chars"] for m in meta]
    print(f"  {cfg}: {len(meta)} pairs")
    if hs:
        print(f"    human chars: min {min(hs)} max {max(hs)} mean {sum(hs)//len(hs)}")
        print(f"    ai chars   : min {min(as_)} max {max(as_)} mean {sum(as_)+0//len(as_)}")
    return len(meta)


if __name__ == "__main__":
    TARGET, FLOOR, N = 150, 100, 31
    print("=== building length-matched arms from HC3-Chinese ===")
    build("medicine", N, TARGET, FLOOR, "hc3_human_medicine", "hc3_ai_medicine")
    build("open_qa", N, TARGET, FLOOR, "hc3_human_openqa", "hc3_ai_openqa")
