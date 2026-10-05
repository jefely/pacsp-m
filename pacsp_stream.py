"""意识流沉积测度 (Stream-of-Consciousness Sedimentation Measure).

The corpus-level measure ``D`` (mean pairwise distance) treats each document as ONE
point and the corpus as an UNORDERED set. This module measures something ``D``
deliberately discards: the order *inside* a single document — the "stream of
consciousness" along which an agent produces its output.

A document is segmented into its natural stream units (paragraphs, falling back to
lines), each unit is embedded, and the sedimentation of the stream is described by
three order-aware, zero-parameter (given the reference frame ``Ω``) quantities:

    S_flow  = mean adjacent distance        how far each step travels
    S_recur = mean nearest-prior distance   how far the stream travels to reach
                                             territory it has already visited
    S_sed   = S_recur / S_flow  in (0, 1]   the sedimentation ratio

Interpretation of S_sed:

    S_sed -> 1 : each unit's nearest earlier unit is (almost) its immediate
                 predecessor; the stream never returns to older content
                 -> pure "flow-through", no re-deposition.
    S_sed -> 0 : the stream repeatedly returns to far-earlier content
                 -> heavy sedimentation (layering onto already-settled sediment).

Why the order here is NOT the arbitrary file order that made C_T sensitive:
C_T summed over the (arbitrary) file order of a corpus. Here the order is the
within-document generation order — the property that "意识流" names. The two are
different objects; only the latter has a claimed physical meaning.

The measure is deliberately minimal so that its declared choices are auditable:

    * segmentation: paragraph blocks (blank-line separated), falling back to lines
      (newline separated) when a document has fewer than ``min_units`` paragraphs;
    * a document with fewer than 2 units has NO defined stream -> metrics are None;
    * a document with exactly 2 units has S_recur == S_flow, so S_sed == 1 by
      construction (degenerate, reported as such, not treated as evidence).

The primary reference frame is inherited from the rest of PACSP-M:
    ``Ω = (R^1024, Euclidean, bge-large-zh-v1.5)``
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Sentence terminators used only as the last-resort fallback; paragraphs and lines
# are preferred so that a "stream unit" is a text block, not a half-sentence.
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])")


def segment_document(text: str, min_units: int = 3) -> list[str]:
    """Split a document into its natural stream units, in generation order.

    Prefers paragraph blocks (blank-line separated); if that yields fewer than
    ``min_units`` units, falls back to non-empty lines. This keeps short poems at
    line granularity and prose at paragraph granularity, deterministically.
    """
    if text is None:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) >= min_units:
        return paras

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= min_units:
        return lines

    # Last resort: sentence-level, for documents written as one unbroken line.
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) >= 2:
        return sents

    return lines if lines else ([text] if text.strip() else [])


def sediment_metrics(embeddings) -> dict:
    """Compute (S_flow, S_recur, S_sed) for an ordered stream of unit embeddings.

    ``embeddings`` is an (n, d) array whose rows follow the generation order of the
    document. Returns a dict; metrics are None when n < 2 (no stream is defined).
    """
    if embeddings is None:
        return {"n": 0, "S_flow": None, "S_recur": None, "S_sed": None}

    emb = np.asarray(embeddings, dtype=float)
    n = emb.shape[0]
    if n < 2:
        return {"n": int(n), "S_flow": None, "S_recur": None, "S_sed": None}

    steps = np.linalg.norm(emb[1:] - emb[:-1], axis=1)  # length n-1
    S_flow = float(steps.mean())

    # nearest-prior distance for each unit k>=1 (min over j<k)
    recur = np.empty(n - 1, dtype=float)
    for k in range(1, n):
        # vectorised distance from emb[k] to every earlier row
        d = np.linalg.norm(emb[:k] - emb[k], axis=1)
        recur[k - 1] = d.min()
    S_recur = float(recur.mean())

    S_sed = float(S_recur / S_flow) if S_flow > 0 else None
    return {
        "n": int(n),
        "S_flow": round(S_flow, 6),
        "S_recur": round(S_recur, 6),
        "S_sed": round(S_sed, 6) if S_sed is not None else None,
    }


def permutation_sediment(embeddings, n_perm: int = 200, seed: int = 11) -> list[float]:
    """Null model: S_sed over random unit-order permutations of the SAME document.

    If the true order carries sedimentation structure (the stream returns to older
    content), the true S_sed sits low in this distribution; if the order is
    information-free, it sits in the middle. Returns the list of permuted S_sed.
    """
    emb = np.asarray(embeddings, dtype=float)
    n = emb.shape[0]
    if n < 3:
        return []
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_perm):
        m = sediment_metrics(emb[rng.permutation(n)])
        if m["S_sed"] is not None:
            out.append(m["S_sed"])
    return out


def document_sediment(text: str, embed) -> dict:
    """Segment + embed + measure one document.

    ``embed`` is a callable mapping a list of strings to an (n, d) array (the same
    interface as ``pacsp_core.compute_embeddings`` wrapped as a partial).
    """
    units = segment_document(text)
    if len(units) == 0:
        return {"n_units": 0, "n": 0, "S_flow": None, "S_recur": None, "S_sed": None}
    emb = embed(units)
    m = sediment_metrics(emb)
    m["n_units"] = len(units)
    return m


def corpus_sediment(directory, embed) -> list[dict]:
    """Measure every ``*.txt`` document in ``directory``; return per-document rows."""
    rows = []
    files = sorted(Path(directory).glob("*.txt"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        m = document_sediment(text, embed)
        m["file"] = f.name
        rows.append(m)
    return rows
