"""Try Hugging Face's newer dataset search endpoints, which index full text.

The plain /api/datasets?search= endpoint returned almost nothing, likely because it
only matches dataset *names*. The newer hub search endpoint matches descriptions
and tags, which is what is needed to find corpora by content.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-search", "Accept": "application/json"}


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.status, r.read()


QUERIES = [
    "AI generated text detection",
    "human AI collaborative writing",
    "machine generated essay detection",
    "human vs machine writing",
    "LLM text detection Chinese",
]

ENDPOINTS = [
    "/api/datasets?search={q}&limit=30&full=true",
    "/api/datasets?search={q}&limit=30",
    "/api/datasets?filter=task_categories:text-classification&search={q}&limit=30",
]

found = {}
for q in QUERIES:
    for tpl in ENDPOINTS:
        url = "https://huggingface.co" + tpl.format(q=urllib.parse.quote(q))
        try:
            code, raw = get(url)
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code}  {tpl[:52]}  q={q!r}")
            continue
        except Exception as e:
            print(f"  FAIL {type(e).__name__}  {tpl[:40]}  q={q!r}")
            continue
        try:
            data = json.loads(raw)
        except Exception:
            print(f"  non-JSON  {tpl[:40]} q={q!r}  {raw[:80]!r}")
            continue
        n = len(data) if isinstance(data, list) else "?"
        print(f"  {code}  {str(n):>3} results  {tpl[:44]}  q={q!r}")
        if isinstance(data, list):
            for d in data:
                i = d.get("id")
                if i:
                    found.setdefault(i, {"id": i,
                                         "downloads": d.get("downloads", 0),
                                         "likes": d.get("likes", 0),
                                         "tags": (d.get("tags") or [])[:8],
                                         "q": q})

print(f"\n=== {len(found)} distinct datasets ===")
rows = sorted(found.values(), key=lambda r: (-(r["downloads"] or 0), -(r["likes"] or 0)))
for r in rows[:40]:
    print(f"  {r['id']:<56} dl={r['downloads']:<7} likes={r['likes']}")
    print(f"      {', '.join(r['tags'])[:100]}")

with open(str(Path(__file__).resolve().parent / "out_hf_candidates2.json"), "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)
print(f"\n  written {len(rows)} to hf_candidates2.json")
