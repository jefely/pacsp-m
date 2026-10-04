"""The measurement core, vendored so this folder stands alone.

PACSP-M imports these from pacsp_build in PACSP-ID, which in turn imports five modules
that exist only for the six-layer attestation stack (pacsp_layer1, pacsp_layer6,
pacsp_merkle, pacsp_sign, pacsp_timestamp). None of that is used by the analyses here,
but the import must succeed, so rather than depend on the sibling repository the six
functions the analyses actually need are copied here.

The function bodies are reproduced character for character from
PACSP-ID/scripts/pacsp_build.py. That matters: this project's standard is that a
reimplementation must be shown numerically identical before it is trusted, and several
comparisons in these documents rest on C_T being the same number here as in the
pipeline. verify_core.py checks it against the original when PACSP-ID is present, and
freezes the expected values so the check still means something when it is not.

Copied from pacsp_build.py:
    load_samples            L42-44
    compute_embeddings      L47-50
    compute_deltas          L53-57
    compute_mus             L60-73
    compute_ct              L76-78
    detect_changepoints     L81-113
"""

from pathlib import Path

import numpy as np


def load_samples(sample_dir):
    files = sorted(Path(sample_dir).glob("*.txt"))
    return [open(f, encoding="utf-8").read() for f in files], files


def compute_embeddings(texts, model_name="BAAI/bge-large-zh-v1.5"):
    """Embed with sentence-transformers, as the pipeline has always done.

    A note on the normalize_embeddings flag, established by measurement rather than assumed.

    The value returned here is unit-norm: measured over the poem corpus the norms are
    1.00000000 with a standard deviation of 3.7e-08, so the flag does not describe the
    vectors that reach compute_deltas and compute_mus.

    The reason is in the model's own modules.json, which composes
        Transformer -> Pooling(pooling_mode_cls_token=True) -> Normalize
    so BAAI/bge-large-zh-v1.5 normalises internally. Passing False suppresses only a second,
    redundant normalisation. Two consequences:

      * the distances in PACSP-M are computed on CLS-pooled, L2-normalised vectors
      * an ONNX reimplementation must reproduce CLS pooling plus L2 normalisation, which is
        what tools/onnx_gpu_acceptance.py verified: max component difference 1.2e-04 against
        sentence-transformers, and the five published D ratios reproduced to 4 decimals

    The flag is therefore redundant for this model, not wrong, and it is left in place so the
    pipeline stays byte-identical to the one that produced the paper.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=8, normalize_embeddings=False)


def compute_deltas(embeddings):
    return [
        float(np.linalg.norm(embeddings[k+1] - embeddings[k]))
        for k in range(len(embeddings) - 1)
    ]


def compute_mus(embeddings, window=5):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-8)
    cos_sim = normalized @ normalized.T

    mus = []
    for k in range(len(embeddings)):
        start = max(0, k - window)
        end = min(len(embeddings), k + window + 1)
        w = cos_sim[start:end, start:end]
        triu = np.triu_indices_from(w, k=1)
        mu = 1 - w[triu].mean() if len(triu[0]) > 0 else 0.0
        mus.append(float(mu))
    return mus


def compute_ct(deltas, mus):
    n = min(len(deltas), len(mus))
    return float(np.sum(np.array(mus[:n]) * np.array(deltas[:n])))


def detect_changepoints(signal, penalty, min_size=5):
    signal = np.asarray(signal, dtype=float)
    n = len(signal)

    if n < min_size * 2:
        return []

    F = np.zeros(n + 1)
    F[1:] = np.inf
    cp = np.zeros(n + 1, dtype=int)

    for t in range(min_size, n + 1):
        best = np.inf
        best_s = 0
        for s in range(0, t - min_size + 1):
            if s > 0 and s < min_size:
                continue
            seg = signal[s:t]
            sse = max(np.sum((seg - seg.mean()) ** 2), 1e-12)
            cost = len(seg) * np.log(sse / len(seg))
            total = F[s] + cost + penalty
            if total < best:
                best = total
                best_s = s
        F[t] = best
        cp[t] = best_s

    cps = []
    t = n
    while cp[t] > 0:
        cps.append(int(cp[t]))
        t = cp[t]
    return sorted(set(cps))


def ct_of_directory(directory, model_name="BAAI/bge-large-zh-v1.5", window=5):
    """Convenience: the C_T of a corpus directory, computed the pipeline's way."""
    texts, files = load_samples(directory)
    emb = compute_embeddings(texts, model_name=model_name)
    deltas = compute_deltas(emb)
    mus = compute_mus(emb, window=window)
    return compute_ct(deltas, mus)
