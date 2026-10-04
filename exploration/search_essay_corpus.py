"""Look for a public corpus of real Chinese student or gaokao essays.

Real human-written essays would give the human arm directly, leaving only the AI arm
to generate and the curated arm for the user to edit -- far less manual work than
writing 31 essays from scratch. A paper on Chinese student essays is referenced in
the search results; this looks for the underlying dataset and for gaokao essay
collections.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-search", "Accept": "application/json"}


def ds_search(term, limit=40):
    url = ("https://huggingface.co/api/datasets?search="
           + urllib.parse.quote(term) + f"&limit={limit}")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:70]}"}


TERMS = ["gaokao", "essay", "student", "zuowen", "作文", "作文", "chinese-essay",
         "student-essay", "argumentative", "writing", "composition"]

seen = {}
for t in TERMS:
    d = ds_search(t)
    if isinstance(d, dict):
        print(f"  {t:<18} ERROR {d.get('err')}")
        continue
    hits = [x for x in d if isinstance(x, dict)]
    print(f"  {t:<18} {len(hits):>3} results")
    for x in hits:
        i = x.get("id")
        if i:
            seen.setdefault(i, {"id": i, "dl": x.get("downloads", 0),
                                "likes": x.get("likes", 0), "term": t,
                                "tags": (x.get("tags") or [])[:5]})

rows = sorted(seen.values(), key=lambda r: (-(r["dl"] or 0), -(r["likes"] or 0)))
KEY = ("gaokao", "essay", "student", "zuowen", "作文", "composition",
       "argumentative", "writing")
rel = [r for r in rows if any(k in r["id"].lower() for k in KEY)]
print(f"\n=== {len(rows)} distinct, {len(rel)} name-relevant ===")
for r in rel[:40]:
    print(f"  {r['id']:<58} dl={r['dl']:<7} likes={r['likes']:<4} via={r['term']}")

with open(str(Path(__file__).resolve().parent / "out_hf_essay_candidates.json"), "w",
          encoding="utf-8") as f:
    json.dump(rel, f, ensure_ascii=False, indent=2)
print(f"\n  written hf_essay_candidates.json")
