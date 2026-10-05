"""情绪树动态区域定位 (Emotion-Tree Dynamic-Region Localisation).

The latest direction: given a document's text content, locate the region of a FIXED
emotion-tree skeleton that the text ACTIVATES, and how that activation moves and
accumulates along the document's stream (its "stream of consciousness").

This is a different object from the corpus measures D (order-invariant dispersion) and
S (semantic recurrence):

    * D / S measure properties of the TEXT ITSELF (its spread / its recurrence).
    * This module measures a RELATION: which region of the emotion tree the text
      activates, and where the activation DEPOSITS as the stream unfolds.

The static skeleton is a DECLARED reference structure (like the reference frame Ω), not
something derived from data. The earlier emotion-tree attempt DERIVED a hierarchy from
C = Y^T Y and was shown to be margin-dominated; here the tree is fixed up front, and the
only job is to project text onto it and localise the activated region.

Projection is by embedding similarity (cosine between a segment embedding and each
emotion word embedding), NOT by a next-token probe phrase. This removes the two
mediators that dominated the earlier attempts: the probe phrase (826x span) and the
emotion-word subset choice. The only declared choices are (1) the tree skeleton, (2) the
embedding space (inherited from the rest of PACSP-M: bge-large-zh-v1.5), and (3) the
temperature used to turn similarities into a distribution (its sensitivity is checked).

Reference frame:  Ω = (R^1024, cosine, bge-large-zh-v1.5).
"""

from __future__ import annotations

import numpy as np

from pacsp_stream import segment_document

# --- 1. The fixed emotion-tree skeleton -------------------------------------------
# 13 basic-emotion prototypes (level 1), each with leaf words (level 2) and a valence
# label. Prototype + leaves together are the same 128-word lexicon used by
# exploration/emotion_tree_probe.py (13 x 10 minus 2 placeholders), re-organised into a
# tree so that "region" is a well-defined subtree.

_VALENCE_POS = "正性"
_VALENCE_NEG = "负性"
_VALENCE_NEU = "中性"

EMOTION_CLUSTERS = [
    ("喜悦", _VALENCE_POS, ["快乐", "高兴", "兴奋", "欣喜", "愉快", "满足", "幸福", "欣慰", "感动"]),
    ("乐观", _VALENCE_POS, ["希望", "期待", "憧憬", "向往", "自信", "骄傲", "自豪", "得意", "轻松"]),
    ("平静", _VALENCE_POS, ["安宁", "宁静", "放松", "释然", "坦然", "安心", "舒畅", "惬意", "从容"]),
    ("爱",   _VALENCE_POS, ["喜爱", "热爱", "眷恋", "思念", "怀念", "牵挂", "温柔", "亲切", "亲密"]),
    ("感激", _VALENCE_POS, ["感谢", "敬佩", "尊敬", "崇拜", "仰慕", "信任", "依赖", "认同", "归属"]),
    ("激励", _VALENCE_POS, ["鼓励", "振奋", "鼓舞", "坚定", "决心", "勇气", "坚韧", "执着", "毅力"]),
    ("悲伤", _VALENCE_NEG, ["难过", "伤心", "痛苦", "悲痛", "哀伤", "凄凉", "忧郁", "沮丧", "失落"]),
    ("孤独", _VALENCE_NEG, ["寂寞", "空虚", "惆怅", "迷茫", "困惑", "无奈", "无力", "绝望", "消沉"]),
    ("愤怒", _VALENCE_NEG, ["生气", "恼怒", "不满", "怨恨", "厌恶", "反感", "嫉妒", "轻蔑", "敌意"]),
    ("恐惧", _VALENCE_NEG, ["害怕", "惊恐", "焦虑", "紧张", "担忧", "不安", "惶恐", "畏惧", "战栗"]),
    ("羞愧", _VALENCE_NEG, ["内疚", "懊悔", "自责", "尴尬", "羞耻", "负罪", "后悔", "遗憾", "惋惜"]),
    ("厌倦", _VALENCE_NEG, ["讽刺", "嘲弄", "冷淡", "漠然", "麻木", "疲惫", "倦怠"]),
    ("惊讶", _VALENCE_NEU, ["震惊", "意外", "诧异", "疑惑", "好奇", "兴趣", "关注", "专注", "投入"]),
]

PROTOTYPES = [p for p, _, _ in EMOTION_CLUSTERS]


def tree_index():
    """Return (words, cluster_of_word, valence_of_word).

    ``cluster_of_word[i]`` is the prototype (level-1 node) that word ``i`` hangs under.
    """
    words, cluster, valence = [], [], []
    for proto, val, leaves in EMOTION_CLUSTERS:
        words.append(proto)
        cluster.append(proto)
        valence.append(val)
        for leaf in leaves:
            words.append(leaf)
            cluster.append(proto)
            valence.append(val)
    return words, cluster, valence


def _cluster_index(cluster):
    """Map each cluster name to the word indices that belong to it."""
    from collections import OrderedDict
    out = OrderedDict()
    for i, c in enumerate(cluster):
        out.setdefault(c, []).append(i)
    return out


def softmax(x, temperature=1.0):
    x = np.asarray(x, dtype=float) / temperature
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# --- 2. Projection ---------------------------------------------------------------


def word_embeddings(embed, words):
    """Embed the emotion words; return an (n_words, d) array."""
    return np.asarray(embed(words), dtype=float)


def project_segment(v, W, temperature=0.1):
    """Activation of one segment over the emotion words.

    Returns a probability distribution a over the words via softmax(cos(v, w)/temp).
    ``temperature`` is a DECLARED parameter; sensitivity to it is checked, not assumed.
    """
    v = np.asarray(v, dtype=float)
    W = np.asarray(W, dtype=float)
    vn = v / (np.linalg.norm(v) + 1e-8)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-8)
    sims = Wn @ vn
    return softmax(sims, temperature=temperature)


def segment_activations(vs, W, temperature=0.1):
    """Activation matrix A (n_segments, n_words) for a document's stream."""
    return np.stack([project_segment(v, W, temperature) for v in vs])


# --- 3. Localisation -------------------------------------------------------------


def document_region(text, embed, W, temperature=0.1):
    """Locate the dynamic activated region of one document.

    Returns a dict with the per-segment trajectory, the cumulative sedimentation, and
    the dominant region. "Sedimentation" is the sum of activations across segments —
    where the stream of consciousness DEPOSITS emotional mass on the tree.
    """
    words, cluster, valence = tree_index()
    cidx = _cluster_index(cluster)

    units = segment_document(text)
    if len(units) == 0:
        return {"n_segments": 0}

    vs = np.asarray(embed(units), dtype=float)
    A = segment_activations(vs, W, temperature=temperature)   # (n_segments, n_words)

    sed = A.sum(axis=0)
    sed = sed / (sed.sum() + 1e-12)                            # cumulative sedimentation

    cluster_sed = {c: float(sed[idx].sum()) for c, idx in cidx.items()}
    dominant = max(cluster_sed, key=cluster_sed.get)

    trajectory = []
    for k in range(A.shape[0]):
        c_sed = {c: float(A[k][idx].sum()) for c, idx in cidx.items()}
        trajectory.append(max(c_sed, key=c_sed.get))

    order = np.argsort(-sed)
    top_words = [(words[i], float(sed[i])) for i in order[:5]]

    return {
        "n_segments": len(units),
        "dominant_cluster": dominant,
        "cluster_sedimentation": cluster_sed,
        "top_words": top_words,
        "trajectory": trajectory,
        "sediment_concentration": float(max(cluster_sed.values())),
    }
