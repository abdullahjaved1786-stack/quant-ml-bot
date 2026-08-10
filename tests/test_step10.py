"""Unit tests for Step 10: News filter (TF-IDF dedup)."""

from __future__ import annotations

import pytest

from src.news_filter import tfidf_dedup, relevance_gate, NewsFilter


# =============================================================================
# tfidf_dedup
# =============================================================================


def test_tfidf_dedup_identical_texts():
    """Identical texts should keep only the first one."""
    texts = ["Bitcoin surges 10%", "Bitcoin surges 10%", "Bitcoin surges 10%"]
    keep = tfidf_dedup(texts, threshold=0.85)
    assert keep == [True, False, False]


def test_tfidf_dedup_distinct_texts():
    """Completely different texts should all be kept."""
    texts = [
        "Bitcoin price surges past 100k",
        "Apple reports record earnings",
        "SpaceX launches new rocket",
    ]
    keep = tfidf_dedup(texts, threshold=0.85)
    assert all(keep)


def test_tfidf_dedup_empty():
    """Empty input → empty output."""
    assert tfidf_dedup([], threshold=0.85) == []


def test_tfidf_dedup_near_duplicate():
    """Slight rewording should be flagged as duplicate."""
    texts = [
        "Ethereum is a decentralized smart contract blockchain platform today",
        "Ethereum is a decentralized smart contract blockchain platform tonight",
    ]
    keep = tfidf_dedup(texts, threshold=0.70)
    assert keep[0] is True
    assert keep[1] is False


def test_tfidf_dedup_preserves_first():
    """The first occurrence should always be kept."""
    texts = ["foo bar baz"] * 5
    keep = tfidf_dedup(texts)
    assert keep[0] is True
    assert sum(keep) == 1


# =============================================================================
# relevance_gate
# =============================================================================


def test_relevance_gate_relevant_text():
    """A text about Bitcoin should be relevant to a Bitcoin reference."""
    ref = "Bitcoin BTC cryptocurrency price"
    texts = ["Bitcoin rallies to new all-time high"]
    keep = relevance_gate(texts, ref, min_similarity=0.1)
    assert keep[0] is True


def test_relevance_gate_irrelevant_text():
    """A text about cooking should not be relevant to Bitcoin."""
    ref = "Bitcoin BTC cryptocurrency price"
    texts = ["How to make the perfect sourdough bread recipe"]
    keep = relevance_gate(texts, ref, min_similarity=0.5)
    assert keep[0] is False


def test_relevance_gate_empty():
    """Empty input → empty output."""
    assert relevance_gate([], "anything") == []


def test_relevance_gate_batch():
    """A batch should return one boolean per item."""
    ref = "Ethereum ETH smart contracts"
    texts = [
        "ETH staking yields rise",
        "New potato variety discovered",
        "Ethereum gas fees drop",
    ]
    keep = relevance_gate(texts, ref, min_similarity=0.1)
    assert len(keep) == 3
    # At least ETH-related items should pass
    assert keep[0] is True or keep[2] is True


# =============================================================================
# NewsFilter class
# =============================================================================


def test_news_filter_dedup_across_batches():
    """Duplicate from a previous batch should be filtered in the next batch."""
    nf = NewsFilter(reference_text="", dedup_threshold=0.85)
    batch1 = ["BTC rallies hard", "ETH staking rises"]
    keep1 = nf.filter_batch(batch1)
    assert keep1 == [True, True]

    batch2 = ["BTC rallies hard", "SOL launches upgrade"]
    keep2 = nf.filter_batch(batch2)
    assert keep2[0] is False  # duplicate
    assert keep2[1] is True   # new


def test_news_filter_relevance_combined():
    """Both relevance and dedup must pass for a text to be kept."""
    nf = NewsFilter(
        reference_text="Bitcoin BTC crypto",
        dedup_threshold=0.85,
        min_relevance=0.05,
    )
    texts = [
        "Bitcoin hits 100k",
        "How to bake chocolate cake",
    ]
    keep = nf.filter_batch(texts)
    assert keep[0] is True
    assert keep[1] is False  # irrelevant


def test_news_filter_empty_batch():
    """Empty batch should return empty list."""
    nf = NewsFilter()
    assert nf.filter_batch([]) == []


def test_news_filter_only_relevant_duplicates_filtered():
    """A relevant duplicate from batch1 should be filtered in batch2."""
    nf = NewsFilter(reference_text="ETH ethereum", dedup_threshold=0.80, min_relevance=0.05)
    nf.filter_batch(["Ethereum price surges 5%"])
    keep = nf.filter_batch(["Ethereum price surges 5%"])
    assert keep == [False]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
