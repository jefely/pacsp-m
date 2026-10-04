"""Search Hugging Face for datasets that could supply a same-source three-arm corpus.

The design needs, for the same task and topic, three conditions:
    human-only  /  human+AI collaborative  /  AI-only
Datasets built for AI-text detection often hold human and machine answers to the
same prompts, and some hold a "human then AI-edited" middle condition, which is
exactly the collaborative arm. This queries the public search API and reports
candidates with size and metadata so they can be assessed rather than guessed at.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-search", "Accept": "application/json"}


def hf(path):
    req = urllib.request.Request("https://huggingface.co" + path, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


QUERIES = [
    "human machine text detection",
    "AI generated text detection",
    "human AI collaborative writing",
    "machine generated essay",
    "human vs llm writing",
    "student essay ai",
    "paraphrase human machine",
    "Chinese human AI text",
]

seen = {}
for q in QUERIES:
    try:
        res = hf("/api/datasets?search=" + urllib.parse.quote(q) + "&limit=15")
    except Exception as e:
        print(f"  query {q!r} failed: {type(e).__name__}")
        continue
    for d in res:
        i = d.get("id")
        if i and i not in seen:
            seen[i] = (q, d)

print(f"=== {len(seen)} distinct dataset candidates ===")
rows = []
for i, (q, d) in seen.items():
    rows.append({
        "id": i,
        "downloads": d.get("downloads", 0),
        "likes": d.get("likes", 0),
        "tags": [t for t in (d.get("tags") or []) if not t.startswith("size_categories")][:6],
        "found_by": q,
    })
rows.sort(key=lambda r: (-(r["downloads"] or 0), -(r["likes"] or 0)))

for r in rows[:30]:
    print(f"  {r['id']:<58} dl={r['downloads']:<7} likes={r['likes']:<4}")
    print(f"      tags: {', '.join(r['tags'])[:96]}")
    print(f"      via : {r['found_by']}")

with open(str(Path(__file__).resolve().parent / "out_hf_candidates.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\n  written {len(rows)} candidates to hf_candidates.json")
