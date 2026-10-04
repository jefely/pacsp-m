"""Inspect the actual schema and splits of the leading candidate corpora.

The three-arm design needs, for one domain and one language:
    human-only  |  human-guided (human curates / edits AI output)  |  AI-only
The middle arm is the scarce one. Parallel human/AI corpora such as HC3 control for
the prompt but give no curated condition; a revision or edit corpus may supply it.

This reads dataset metadata and a few rows, so claims about content come from the
data rather than from the dataset name.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "dsh-corpus-inspect", "Accept": "application/json"}


def get(url, accept="application/json"):
    req = urllib.request.Request(url, headers={"User-Agent": UA["User-Agent"],
                                               "Accept": accept})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def meta(ds):
    try:
        return json.loads(get(f"https://huggingface.co/api/datasets/{ds}"))
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:90]}"}


def rows(ds, config=None, split="train", limit=3, offset=0):
    url = ("https://datasets-server.huggingface.co/rows?dataset="
           + urllib.parse.quote(ds) + f"&split={split}&offset={offset}&length={limit}")
    if config:
        url += "&config=" + urllib.parse.quote(config)
    try:
        return json.loads(get(url))
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}


def splits(ds):
    url = ("https://datasets-server.huggingface.co/splits?dataset="
           + urllib.parse.quote(ds))
    try:
        return json.loads(get(url))
    except Exception as e:
        return {"err": f"{type(e).__name__}: {str(e)[:120]}"}


CANDS = [
    "Hello-SimpleAI/HC3-Chinese",
    "Hello-SimpleAI/HC3",
    "yaful/MAGE",
    "aimagelab/RAID",
    "liamdugan/raid",
    "jiancui/mage",
    "alexshpunt/explicit-edit-benchmark",
]

for ds in CANDS:
    print("=" * 74)
    print(ds)
    m = meta(ds)
    if "err" in m:
        print(f"  metadata error: {m['err']}")
        continue
    print(f"  downloads={m.get('downloads')} likes={m.get('likes')}")
    print(f"  tags: {', '.join((m.get('tags') or [])[:10])}")
    cfgs = m.get("configs") or []
    print(f"  configs: {len(cfgs)}")
    for c in cfgs[:6]:
        print(f"    - {c.get('config_name')}: splits={[s.get('split') for s in (c.get('splits') or [])]}")

    sp = splits(ds)
    if isinstance(sp, dict) and sp.get("splits"):
        print(f"  splits endpoint:")
        for s in sp["splits"][:8]:
            print(f"    config={s.get('config')} split={s.get('split')} "
                  f"rows={s.get('num_rows')}")
        cfg = sp["splits"][0].get("config")
        split = sp["splits"][0].get("split")
    else:
        cfg, split = None, "train"

    r = rows(ds, cfg, split, limit=2)
    if isinstance(r, dict) and r.get("rows"):
        print(f"  sample columns: {list(r['rows'][0]['row'].keys())}")
        for rw in r["rows"][:2]:
            for k, v in rw["row"].items():
                s = str(v)
                print(f"      {k:<22} {s[:110]}")
            print("      ---")
    else:
        err = r.get("err") if isinstance(r, dict) else str(r)[:100]
        print(f"  rows unavailable: {err}")
    print()
