"""Work out how to get prose out of deepseek-r1 within a usable budget.

The first attempt spent the entire output budget (num_predict 4000) inside an
unclosed <think> block, so the essay never started. Options: raise the budget a lot,
or stream and stop once the thinking block closes. This measures how much budget the
thinking actually consumes on a real prompt.
"""

import json
import re
import socket
import time
import urllib.error
import urllib.request

socket.setdefaulttimeout(1800)
OLLAMA = "http://127.0.0.1:11434"
MODEL = "deepseek-r1:14b"

PROMPT = ("请以高考作文的规范写一篇不少于800字的议论文。题目材料如下：\n\n"
          "人们因技术发展得以更好地掌控时间，但也有人因此成了时间的仆人。"
          "这句话引发了你怎样的联想与思考？请写一篇文章。\n\n"
          "要求：自拟标题；立意明确，结构完整，论证充分。"
          "直接输出文章正文，不要输出任何解释、前言或评分说明。")


def stream_gen(prompt, seed, num_predict):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.8, "top_p": 0.9,
                    "num_predict": num_predict, "seed": seed},
    }
    req = urllib.request.Request(OLLAMA + "/api/generate",
                                 data=json.dumps(payload).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    chunks = []
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("response"):
                chunks.append(d["response"])
            if d.get("done"):
                break
    return "".join(chunks), time.time() - t0


for budget in (4000, 12000):
    print(f"=== streaming, num_predict={budget} ===")
    try:
        text, dt = stream_gen(PROMPT, 13, budget)
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {str(e)[:120]}")
        continue
    has_open = "<think>" in text
    has_close = "</think>" in text
    print(f"  total chars   : {len(text):,}   ({dt:.0f}s)")
    print(f"  <think> open  : {has_open}")
    print(f"  </think> close: {has_close}")
    if has_close:
        head, tail = text.split("</think>", 1)
        print(f"  thinking chars: {len(head):,}")
        print(f"  answer chars  : {len(tail.strip()):,}")
        print(f"  answer head   : {tail.strip()[:150]!r}")
    else:
        print(f"  tail          : {text[-120:]!r}")
    print()
