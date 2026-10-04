"""Count what HC3-Chinese actually offers per domain, and check text lengths.

Two-arm corpora give the prompt-parallel contrast: for each question there is a
human answer and a ChatGPT answer. That is a real, citable control for authorship,
but it is NOT the paper's middle arm (human curates AI output), which no public
corpus found so far provides.

Sizes matter because the existing PACSP arms are 31 items each; if a domain has
enough questions, 31 paired items can be drawn from it directly.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-count"}


def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={**UA, "Accept": accept})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


DS = "Hello-SimpleAI/HC3-Chinese"
DOMAINS = ["all", "baike", "finance", "law", "medicine", "nlpcc_dbqa",
           "open_qa", "psychology"]


def rows(cfg, offset, length):
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           + urllib.parse.quote(DS)
           + f"&config={urllib.parse.quote(cfg)}&split=train"
           + f"&offset={offset}&length={length}")
    return json.loads(get(url))


print("=== per-domain availability ===")
for cfg in DOMAINS:
    try:
        r = rows(cfg, 0, 1)
    except urllib.error.HTTPError as e:
        print(f"  {cfg:<14} HTTP {e.code}")
        continue
    except Exception as e:
        print(f"  {cfg:<14} {type(e).__name__}: {str(e)[:60]}")
        continue
    n = r.get("num_rows_total")
    cols = list((r.get("rows") or [{}])[0].get("row", {}).keys())
    print(f"  {cfg:<14} rows={n}  cols={cols}")

print()
print("=== sample lengths in a candidate domain (finance) ===")
try:
    r = rows("finance", 0, 8)
    for rw in r.get("rows", []):
        row = rw["row"]
        q = row.get("question") or ""
        h = row.get("human_answers") or []
        c = row.get("chatgpt_answers") or []
        hl = sum(len(x) for x in h if isinstance(x, str))
        cl = sum(len(x) for x in c if isinstance(x, str))
        print(f"  q={len(q):>4}c  human={len(h)}ans/{hl:>6}c  gpt={len(c)}ans/{cl:>6}c  "
              f"| {q[:44]}")
except Exception as e:
    print(f"  failed: {type(e).__name__}: {e}")
