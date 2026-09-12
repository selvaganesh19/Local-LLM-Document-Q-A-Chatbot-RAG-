"""Tests for the semantic response cache."""

from __future__ import annotations

import time

import pytest

from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.ingestion.embedder import HashingEmbedder

PAYLOAD = {"answer": "90 calendar days", "citations": [1]}


@pytest.fixture
def cache(settings: Settings, embedder: HashingEmbedder) -> SemanticCache:
    """A cache using the deterministic test embedder."""
    return SemanticCache(embedder=embedder, settings=settings)


def test_exact_query_hits(cache: SemanticCache) -> None:
    """The same question returns the stored payload."""
    cache.set("What is the notice period?", PAYLOAD)

    hit = cache.get("What is the notice period?")

    assert hit is not None
    assert hit.payload["answer"] == "90 calendar days"
    assert hit.similarity == pytest.approx(1.0)
    assert hit.matched_query == "What is the notice period?"


def test_reordered_query_hits_with_the_bag_of_words_embedder(cache: SemanticCache) -> None:
    """A rephrasing with identical vocabulary still matches."""
    cache.set("notice period senior staff", PAYLOAD)

    hit = cache.get("senior staff notice period")

    assert hit is not None
    assert hit.similarity >= 0.9


def test_unrelated_query_misses(cache: SemanticCache) -> None:
    """A question with different vocabulary is not served from cache."""
    cache.set("What is the notice period?", PAYLOAD)

    assert cache.get("How do I rotate an API token?") is None


def test_empty_query_misses(cache: SemanticCache) -> None:
    """Blank queries are never stored or matched."""
    cache.set("   ", PAYLOAD)

    assert cache.get("") is None
    assert len(cache) == 0


def test_disabled_cache_returns_nothing(settings: Settings, embedder: HashingEmbedder) -> None:
    """With caching switched off the cache is transparent."""
    disabled = settings.model_copy(update={"cache_enabled": False})
    cache = SemanticCache(embedder=embedder, settings=disabled)

    cache.set("question", PAYLOAD)

    assert cache.get("question") is None
    assert cache.enabled is False


def test_expired_entry_is_not_served(settings: Settings, embedder: HashingEmbedder) -> None:
    """Entries past their TTL are dropped on the next lookup."""
    short_ttl = settings.model_copy(update={"cache_ttl_seconds": 1})
    cache = SemanticCache(embedder=embedder, settings=short_ttl)
    cache.set("question", PAYLOAD)

    # Age the entry rather than sleeping, so the test stays fast and stable.
    cache._entries[0].created_at = time.time() - 10  # noqa: SLF001 - deliberate ageing

    assert cache.get("question") is None


def test_cache_evicts_least_recently_used(settings: Settings, embedder: HashingEmbedder) -> None:
    """The cache stays within its configured size bound."""
    small = settings.model_copy(update={"cache_max_entries": 3})
    cache = SemanticCache(embedder=embedder, settings=small)

    for index in range(6):
        cache.set(f"unique question about token{index}", {"answer": f"answer{index}"})

    assert len(cache) <= 3
    stats = cache.stats()
    assert stats["entries"] <= stats["max_entries"]


def test_inserting_a_near_duplicate_replaces_rather_than_appends(cache: SemanticCache) -> None:
    """Paraphrases of one question do not fill the cache with duplicates."""
    cache.set("what is the notice period", {"answer": "first"})
    cache.set("what is the notice period", {"answer": "second"})

    assert len(cache) == 1
    hit = cache.get("what is the notice period")
    assert hit is not None
    assert hit.payload["answer"] == "second"


def test_stats_track_hits_and_misses(cache: SemanticCache) -> None:
    """Counters reflect observed traffic."""
    cache.set("what is the notice period", PAYLOAD)
    cache.get("what is the notice period")
    cache.get("completely unrelated question")

    stats = cache.stats()

    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(0.5)
    assert stats["enabled"] is True


def test_clear_removes_entries_and_counters(cache: SemanticCache) -> None:
    """Clearing empties the cache and resets its statistics."""
    cache.set("what is the notice period", PAYLOAD)
    cache.get("what is the notice period")

    removed = cache.clear()

    assert removed == 1
    assert len(cache) == 0
    assert cache.stats()["hits"] == 0
    assert cache.stats()["misses"] == 0


def test_hits_are_counted_on_the_entry(cache: SemanticCache) -> None:
    """Repeated hits increment the entry's own counter."""
    cache.set("what is the notice period", PAYLOAD)

    first = cache.get("what is the notice period")
    second = cache.get("what is the notice period")

    assert first is not None and second is not None
    assert first.original_hits == 0
    assert second.original_hits == 1
