"""Check whether pacsp_core's embeddings are unit-norm despite normalize_embeddings=False.

The ONNX parity test selected CLS pooling with normalisation as the match, at max difference
1.2e-4, while CLS without normalisation differed by 7.4. That is a large enough gap to mean
the reference vectors carry unit norm regardless of the flag, which would make the flag
misleading rather than effective.

This is worth settling because the paper's numbers depend on the embedding actually used, and
because a future reader will trust the flag. If the vectors are unit-norm, the comment in
pacsp_core should say so.
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
M = Path(r"D:\myproject\PACSP-M")
CACHE = M / "_hf_home"
sys.path.insert(0, str(M))
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

MODEL_ID = "BAAI/bge-large-zh-v1.5"


def main():
    import pacsp_core
    texts = pacsp_core.load_samples(M / "data" / "poem")[0]
    E = np.asarray(pacsp_core.compute_embeddings(texts), dtype=np.float64)
    norms = np.linalg.norm(E, axis=1)
    print(f"  texts        : {len(texts)}")
    print(f"  shape        : {E.shape}")
    print(f"  norm min     : {norms.min():.8f}")
    print(f"  norm max     : {norms.max():.8f}")
    print(f"  norm mean    : {norms.mean():.8f}")
    print(f"  norm std     : {norms.std():.2e}")
    unit = bool(np.allclose(norms, 1.0, atol=1e-5))
    print(f"\n  all unit norm: {unit}")

    print("\n  compare against SentenceTransformer with each flag:")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(MODEL_ID)
    for flag in (False, True):
        S = np.asarray(st.encode(texts, batch_size=32,
                                 normalize_embeddings=flag), dtype=np.float64)
        sn = np.linalg.norm(S, axis=1)
        d = float(np.abs(S - E).max())
        print(f"    normalize_embeddings={str(flag):<5} norm mean {sn.mean():.6f}  "
              f"maxdiff vs pacsp_core {d:.3e}")

    print("\n  conclusion:")
    if unit:
        print("    pacsp_core returns unit-norm embeddings even though it passes")
        print("    normalize_embeddings=False. The flag therefore does not describe what")
        print("    the pipeline receives; the vectors are CLS pooled and L2 normalised.")
        print("    This should be stated in pacsp_core, because every distance in the")
        print("    paper is computed on these vectors and a reader would otherwise")
        print("    assume unnormalised ones.")
    else:
        print("    The vectors are not unit norm; the earlier pooling match needs")
        print("    re-examination.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
