"""Tests for length-bucketed (DDP-shardable) batching."""

from __future__ import annotations

from pm_foundation.data.samplers import LengthBucketedSampler


def test_covers_all_indices_exactly_once() -> None:
    lengths = [i % 30 + 1 for i in range(205)]
    sampler = LengthBucketedSampler(lengths, batch_size=16, shuffle=True, seed=0)
    batches = list(sampler)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(205))  # every index exactly once
    assert len(batches) == len(sampler)


def test_batches_are_length_homogeneous() -> None:
    # A small tail of long sequences must not inflate every batch.
    lengths = [5] * 500 + [200] * 20
    sampler = LengthBucketedSampler(lengths, batch_size=32, shuffle=True, pool_factor=50, seed=0)
    padded = sum(max(lengths[i] for i in b) * len(b) for b in sampler)
    ideal = sum(lengths)
    naive_worst = 200 * len(lengths)
    assert padded < 0.25 * naive_worst  # bucketing keeps padded work near the ideal
    assert padded < 3 * ideal


def test_no_shuffle_is_globally_sorted() -> None:
    lengths = [7, 1, 5, 3, 9, 2, 8, 4]
    sampler = LengthBucketedSampler(lengths, batch_size=3, shuffle=False)
    ordered = [lengths[i] for b in sampler for i in b]
    assert ordered == sorted(lengths)


def test_drop_last() -> None:
    sampler = LengthBucketedSampler([1] * 10, batch_size=4, shuffle=False, drop_last=True)
    batches = list(sampler)
    assert all(len(b) == 4 for b in batches)
    assert len(batches) == 2 == len(sampler)


def test_ddp_shards_are_disjoint_and_equal() -> None:
    lengths = [i % 40 + 1 for i in range(1000)]
    world = 4
    per_rank = [
        list(LengthBucketedSampler(lengths, 16, shuffle=True, num_replicas=world, rank=r, seed=7))
        for r in range(world)
    ]
    # Equal number of batches per rank (DDP requires matched step counts).
    counts = {len(b) for b in per_rank}
    assert len(counts) == 1
    assert all(
        len(b) == len(LengthBucketedSampler(lengths, 16, num_replicas=world, rank=r, seed=7))
        for r, b in enumerate(per_rank)
    )
    # Disjoint indices across ranks (no sample trained twice in one step-set).
    seen: set[int] = set()
    for batches in per_rank:
        idx = {i for b in batches for i in b}
        assert seen.isdisjoint(idx)
        seen |= idx


def test_ddp_reshuffles_by_epoch_consistently() -> None:
    lengths = [i % 20 + 1 for i in range(200)]
    a = LengthBucketedSampler(lengths, 8, shuffle=True, num_replicas=2, rank=0, seed=1)
    b = LengthBucketedSampler(lengths, 8, shuffle=True, num_replicas=2, rank=0, seed=1)
    a.set_epoch(3)
    b.set_epoch(3)
    assert list(a) == list(b)  # same epoch+seed -> identical (so ranks agree)
    a.set_epoch(4)
    assert list(a) != list(b)  # different epoch -> reshuffled
