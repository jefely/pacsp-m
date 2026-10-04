"""Verify the push landed completely, through the API rather than through git.

git cannot reach GitHub in this environment except with the openssl backend passed
explicitly, so the check reads the remote ref and tree over the REST API instead. This
confirms the branch is at the expected commit and that the tracked files arrived.
"""

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(r"D:\myproject\PACSP-M")
# Path from argv so this cannot drift from the step that wrote the token file.
_tk = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    r"D:\myproject\chrome-debug\_gh_token.txt")
TOKEN = _tk.read_text(encoding="utf-8").strip()
OWNER, NAME = "jefely", "pacsp-m"
API = "https://api.github.com"


def api(path):
    req = urllib.request.Request(
        API + path,
        headers={"Authorization": f"Bearer {TOKEN}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "pacsp-m-verify"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:300]}
    except Exception as e:
        return 0, {"error": f"{type(e).__name__}: {e}"}


def local(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, timeout=300)
    return r.returncode, (r.stdout or b"").decode("utf-8", "replace").strip()


def main():
    rc, local_head = local("rev-parse", "HEAD")
    rc2, local_count = local("rev-list", "--count", "HEAD")
    rc3, tracked = local("ls-files")
    n_tracked = len(tracked.splitlines())

    print(f"  local HEAD        : {local_head}")
    print(f"  local commits     : {local_count}")
    print(f"  local tracked file: {n_tracked}")

    st, ref = api(f"/repos/{OWNER}/{NAME}/git/ref/heads/main")
    remote_sha = ref.get("object", {}).get("sha") if st == 200 else None
    print(f"\n  GET ref/heads/main -> {st}")
    print(f"  remote HEAD       : {remote_sha}")
    print(f"  match             : {remote_sha == local_head}")

    if remote_sha:
        st2, tree = api(f"/repos/{OWNER}/{NAME}/git/trees/{remote_sha}?recursive=1")
        if st2 == 200:
            entries = tree.get("tree", [])
            blobs = [e for e in entries if e["type"] == "blob"]
            print(f"\n  remote tree       : {len(entries)} entries, "
                  f"{len(blobs)} blobs, truncated={tree.get('truncated')}")
            total = sum(e.get("size", 0) for e in blobs)
            print(f"  remote bytes      : {total:,}")
            for want in ("PACSP-M-1.1.0.md", "pacsp_core.py", "verify_paper_numbers.py",
                         "dist/PACSP-M-1.1.0.pdf", "dist/PACSP-M-1.1.0.docx",
                         "README.md", "results"):
                hit = [e for e in entries if e["path"] == want
                       or e["path"].startswith(want + "/")]
                print(f"    {want:<32} {'present' if hit else 'MISSING'}"
                      f"{f'  ({len(hit)} files)' if len(hit) > 1 else ''}")

    st3, repo = api(f"/repos/{OWNER}/{NAME}")
    if st3 == 200:
        print(f"\n  repo              : {repo.get('html_url')}")
        print(f"  description       : {repo.get('description')}")
        print(f"  default branch    : {repo.get('default_branch')}")
        print(f"  visibility        : {'private' if repo.get('private') else 'public'}")

    rc4, remotes = local("remote", "-v")
    print(f"\n  local remote config (must carry no credential):")
    for line in remotes.splitlines():
        safe = line
        for marker in ("gho_", "ghp_", "ghs_"):
            if marker in safe:
                i = safe.index(marker)
                safe = safe[:i] + "***REDACTED***"
        print(f"    {safe}")

    return 0 if remote_sha == local_head else 1


if __name__ == "__main__":
    sys.exit(main())
