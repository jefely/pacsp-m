"""Report a repository's remote HEAD without relying on local git credentials.

Why this is needed
------------------
`git log --oneline origin/HEAD..HEAD | Measure-Object).Count` reads as "how many commits are
unpushed", but when `origin/HEAD` does not resolve the range is empty and the count comes back
**0** — a false "everything is pushed". That is how PACSP-收集 was reported as archived when a
local credential misconfiguration meant it could not have been pushed at all.

This asks GitHub directly, so the answer does not depend on the local clone's state.

Usage:  python tools/check_remote_head.py <owner/repo> [<owner/repo> ...]
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_visibility_module():
    spec = importlib.util.spec_from_file_location("_vis", HERE / "set_repo_visibility.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # module body only defines things
    return mod


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    m = load_visibility_module()
    token = m.read_token()
    rc = 0
    for slug in sys.argv[1:]:
        st, repo = m.call(f"/repos/{slug}", token)
        if st != 200:
            print(f"  [X] {slug}: HTTP {st}  {str(repo.get('error', repo))[:150]}")
            rc = 1
            continue
        vis = "private" if repo.get("private") else "public"
        st2, br = m.call(f"/repos/{slug}/branches/{repo['default_branch']}", token)
        print(f"  {slug}")
        print(f"    可见性        : {vis}")
        print(f"    默认分支      : {repo.get('default_branch')}")
        if st2 == 200:
            c = br["commit"]["commit"]
            print(f"    远程 HEAD     : {br['commit']['sha'][:7]}  "
                  f"{c['message'].splitlines()[0][:64]}")
            print(f"    提交时间      : {c['committer']['date']}")
        else:
            print(f"    远程 HEAD     : HTTP {st2}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
