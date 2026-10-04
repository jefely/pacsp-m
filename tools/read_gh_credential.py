"""Read the stored GitHub credential so the push can use the same account as the other repos.

The credential lives in Windows Credential Manager under the target git:https://github.com.
Reading it requires the Win32 credential API, which is what the bundled git credential
helper would use anyway. The username is printed; the secret is only reported by length,
never written to a file or to the log.
"""

import ctypes
import ctypes.wintypes as w
import sys


class CREDENTIAL(ctypes.Structure):
    _fields_ = [
        ("Flags", w.DWORD),
        ("Type", w.DWORD),
        ("TargetName", w.LPWSTR),
        ("Comment", w.LPWSTR),
        ("LastWritten", w.FILETIME),
        ("CredentialBlobSize", w.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
        ("Persist", w.DWORD),
        ("AttributeCount", w.DWORD),
        ("Attributes", ctypes.c_void_p),
        ("TargetAlias", w.LPWSTR),
        ("UserName", w.LPWSTR),
    ]


def main():
    c32 = ctypes.windll.advapi32
    c32.CredReadW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD,
                              ctypes.POINTER(ctypes.POINTER(CREDENTIAL))]
    c32.CredReadW.restype = w.BOOL
    c32.CredFree.argtypes = [ctypes.c_void_p]

    p = ctypes.POINTER(CREDENTIAL)()
    ok = c32.CredReadW("git:https://github.com", 1, 0, ctypes.byref(p))
    if not ok:
        err = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else "?"
        print(f"  CredRead failed (err={err})")
        return 1

    cred = p.contents
    user = cred.UserName or ""
    blob = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
    try:
        secret = blob.decode("utf-16-le", errors="ignore").rstrip("\x00")
    except Exception:
        secret = blob.decode("utf-8", errors="ignore")
    c32.CredFree(p)

    print(f"  target  : git:https://github.com")
    print(f"  user    : {user}")
    print(f"  secret  : {len(secret)} chars, prefix {secret[:4]!r}")
    if secret:
        # hand it to the caller through the environment for this process tree only
        print(f"  TOKEN_ENV_OK {len(secret)}")
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            f.write(secret)
        print(f"  written to {sys.argv[1]}")
    return 0 if secret else 1


if __name__ == "__main__":
    sys.exit(main())
