"""News/signal filter: TF-IDF dedup and relevance gating.

Filters duplicate or near-duplicate news items using cosine similarity
over TF-IDF vectors. Also provides a relevance gate that rejects items
below a minimum similarity to the asset's description.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_dedup(
    texts: list[str],
    threshold: float = 0.85,
) -> list[bool]:
    """Mark duplicate texts using TF-IDF cosine similarity.

    Args:
        texts: List of text strings to deduplicate.
        threshold: Cosine similarity above which two items are considered
                   duplicates (default 0.85).

    Returns:
        Boolean list: True = keep, False = duplicate of an earlier item.
    """
    if len(texts) == 0:
        return []

    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    keep = [True] * len(texts)
    for i in range(1, len(texts)):
        sim = cosine_similarity(X[i], X[:i]).max()
        if float(sim) >= threshold:
            keep[i] = False
    return keep


def relevance_gate(
    texts: list[str],
    reference_text: str,
    min_similarity: float = 0.1,
) -> list[bool]:
    """Filter texts by relevance to a reference (e.g., asset description).

    Args:
        texts: List of candidate text strings.
        reference_text: Reference string (e.g., asset name/description).
        min_similarity: Minimum cosine similarity to pass (default 0.1).

    Returns:
        Boolean list: True = relevant enough, False = not relevant.
    """
    if not texts:
        return []

    corpus = [reference_text] + texts
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(corpus)
    ref_vec = X[0:1]
    cand_vecs = X[1:]

    sims = cosine_similarity(cand_vecs, ref_vec).ravel()
    return [bool(s >= min_similarity) for s in sims]


class NewsFilter:
    """Filter a stream of news items: dedup then relevance gate.

    Args:
        reference_text: Reference text for relevance (e.g., asset description).
        dedup_threshold: Cosine similarity above which items are duplicates.
        min_relevance: Minimum similarity to reference_text to pass.
    """

    def __init__(
        self,
        reference_text: str = "",
        dedup_threshold: float = 0.85,
        min_relevance: float = 0.1,
    ):
        self.reference_text = reference_text
        self.dedup_threshold = dedup_threshold
        self.min_relevance = min_relevance
        self._seen_texts: list[str] = []

    def filter_batch(self, texts: list[str]) -> list[bool]:
        """Dedup against previously seen texts + new batch, then relevance gate.

        Args:
            texts: List of text strings to filter.

        Returns:
            Boolean list: True = pass, False = filtered out.
        """
        if not texts:
            return []

        # Relevance gate (fast; no memory)
        if self.reference_text:
            rel_mask = relevance_gate(
                texts, self.reference_text, min_similarity=self.min_relevance
            )
        else:
            rel_mask = [True] * len(texts)

        # Dedup gate (requires accumulated context)
        all_texts = self._seen_texts + texts
        dedup_mask = tfidf_dedup(all_texts, threshold=self.dedup_threshold)
        new_dedup = dedup_mask[len(self._seen_texts):]

        # Update seen texts
        for text, keep in zip(texts, new_dedup):
            if keep:
                self._seen_texts.append(text)

        # Both must pass
        return [r and d for r, d in zip(rel_mask, new_dedup)]
