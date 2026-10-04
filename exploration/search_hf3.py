"""Look for corpora by known dataset names, and settle what a collaborative arm is.

Two questions:
  1. which public corpora offer human-written and machine-written answers to the
     SAME prompts (a parallel human / AI pair)
  2. is there a natural "human then AI-edited" middle condition anywhere, or does
     the collaborative arm have to be constructed

The middle condition matters more than it looks. The paper's own measurement is that
between-domain differences (6.09x across arms) dwarf between-axis differences
(0.33-2.78x), so an arm has to differ from its partners by as little as possible
apart from the variable under test. A corpus that pairs a human answer with an
AI *edit of that same answer* controls for topic and prompt exactly; two
independently generated answers from the same prompt control for prompt only.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-search", "Accept": "application/json"}


def search(term, limit=50):
    url = ("https://huggingface.co/api/datasets?search="
           + urllib.parse.quote(term) + f"&limit={limit}")
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:70]}"}


TERMS = ["hc3", "m4gt", "semEval", "raid", "mage", "aigc", "chatgpt",
         "gptzero", "detectgpt", "ghostbuster", "edit", "rewrite",
         "paraphrase", "collaborat", "assist", "revision", "coauthor",
         "Chinese", "chinese", "zh", "essay", "abstract", "review"]

found = {}
for t in TERMS:
    d = search(t)
    if isinstance(d, dict):
        print(f"  {t:<14} ERROR")
        continue
    hits = [x for x in d if isinstance(x, dict)]
    print(f"  {t:<14} {len(hits):>3} results")
    for x in hits:
        i = x.get("id")
        if i:
            found.setdefault(i, {
                "id": i, "downloads": x.get("downloads", 0),
                "likes": x.get("likes", 0), "term": t,
                "tags": (x.get("tags") or [])[:6]})

print(f"\n=== {len(found)} distinct datasets ===")
rows = sorted(found.values(), key=lambda r: (-(r["downloads"] or 0), -(r["likes"] or 0)))
# prioritise names that suggest human/machine contrast or editing
KEY = ("hc3", "m4gt", "mage", "aigc", "raid", "detect", "machine", "human",
       "chatgpt", "gpt", "llm", "ai-", "_ai", "paraphrase", "edit", "rewrite",
       "collab", "assist", "semeval")
interesting = [r for r in rows
               if any(k in r["id"].lower() for k in KEY)]
print(f"  ({len(interesting)} look potentially relevant)\n")
for r in interesting[:45]:
    print(f"  {r['id']:<60} dl={r['downloads']:<8} likes={r['likes']:<4} via={r['term']}")

with open(str(Path(__file__).resolve().parent / "out_hf_candidates3.json"), "w", encoding="utf-8") as f:
    json.dump({"all": rows, "interesting": interesting}, f,
              ensure_ascii=False, indent=2)
print(f"\n  written hf_candidates3.json ({len(rows)} total, {len(interesting)} interesting)")
