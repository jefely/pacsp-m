"""Check that the page now renders the assignability row and labels it as the criterion.

The row exists in the payload either way; what matters is whether the HTML the browser gets
contains it and marks it as what decides the verdict, since the point of adding it was that
overlap was being read as if it decided.
"""

import json
import sys
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8731"
M = Path(r"D:\myproject\PACSP-M")


def main():
    with urllib.request.urlopen(BASE + "/", timeout=30) as r:
        page = r.read().decode("utf-8")
    print(f"  page: {len(page)} bytes")

    need = [
        ("assignability is read from the payload", "c.assignability"),
        ("labelled as the criterion", "判定依据"),
        ("labelled as not the criterion", "不是判据"),
        ("the inverse-correlation note is present", "−0.90"),
        ("the table header says what decides", "判定使用"),
        ("colouring classes exist", "v-ok"),
    ]
    bad = []
    for label, needle in need:
        ok = needle in page
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}")
        if not ok:
            bad.append(label)

    # and the payload really carries the field the row reads
    payload = {"a": str(M / "data" / "poem"),
               "b": str(M / "data" / "machine_poem"),
               "frame": "bge-large-zh", "backend": "onnx", "bootstrap": 500}
    req = urllib.request.Request(BASE + "/api/run", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    c = d["compare"]
    has = "assignability" in c
    print(f"  [{'ok  ' if has else 'FAIL'}] payload has assignability: "
          f"{c.get('assignability')}")
    if not has:
        bad.append("payload missing assignability")

    print()
    if bad:
        print(f"  {len(bad)} PROBLEM(S): {bad}")
        return 1
    print("  the interface now shows the criterion it judges by")
    return 0


if __name__ == "__main__":
    sys.exit(main())
