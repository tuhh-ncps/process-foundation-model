"""Batch samplers for variable-length traces.

Padding a batch to its longest member wastes compute — badly when a small tail of long
traces drags nearly every random batch up to the max (see docs §18). ``LengthBucketedSampler``
groups similar-length traces into the same batch (each batch pads only to its own small max)
while a pooled-shuffle scheme keeps epoch-to-epoch randomness.

It is also **DDP-aware**: with ``num_replicas``/``rank`` set, every rank builds the *same*
epoch-seeded batch list and then takes a disjoint stride of batches, with the count truncated
to a multiple of ``num_replicas`` so all ranks run the same number of steps (a DDP requirement).
Because a custom ``batch_sampler`` disables Lightning's automatic ``DistributedSampler``, this
class is what shards the data across ranks.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler

_PERM_OFFSET = 0x9E3779B9  # decorrelate the batch-order shuffle from the index shuffle


class LengthBucketedSampler(Sampler[list[int]]):
    """Yield batches of indices grouped by length (pooled megabatch sort), DDP-shardable.

    Each epoch: (optionally) shuffle all indices with a seed shared across ranks, cut into
    pools of ``pool_factor·batch_size``, sort each pool by length, cut into batches, shuffle
    the batch order, then keep this rank's stride. ``shuffle=False`` is a deterministic global
    length sort (ideal for eval loaders). Call :meth:`set_epoch` each epoch to reshuffle.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        *,
        shuffle: bool = True,
        pool_factor: int = 50,
        drop_last: bool = False,
        generator: torch.Generator | None = None,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not 0 <= rank < max(1, num_replicas):
            raise ValueError(f"rank {rank} out of range for num_replicas {num_replicas}")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pool = max(1, pool_factor) * batch_size
        self.drop_last = drop_last
        self.generator = generator  # single-process reproducibility (unused under DDP)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch so ``shuffle`` reseeds identically across ranks (called by Lightning)."""
        self.epoch = epoch

    def _shuffle_generator(self, salt: int) -> torch.Generator | None:
        # Under DDP every rank must shuffle identically, so derive from seed+epoch. Single
        # process keeps the caller's generator (or the global RNG) — unchanged local behaviour.
        if self.num_replicas > 1:
            return torch.Generator().manual_seed(self.seed + self.epoch + salt)
        return self.generator

    def _all_batches(self) -> list[list[int]]:
        n = len(self.lengths)
        if self.shuffle:
            order = torch.randperm(n, generator=self._shuffle_generator(0)).tolist()
        else:
            order = list(range(n))
        batches: list[list[int]] = []
        for start in range(0, n, self.pool):
            pool = order[start : start + self.pool]
            pool.sort(key=lambda j: self.lengths[j])  # similar lengths land together
            for b in range(0, len(pool), self.batch_size):
                batch = pool[b : b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        if self.shuffle:
            gen = self._shuffle_generator(_PERM_OFFSET)
            batches = [batches[k] for k in torch.randperm(len(batches), generator=gen).tolist()]
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._all_batches()
        if self.num_replicas > 1:
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[self.rank : usable : self.num_replicas]  # disjoint per-rank stride
        yield from batches

    def __len__(self) -> int:
        total, n = 0, len(self.lengths)
        for start in range(0, n, self.pool):
            p = min(self.pool, n - start)
            total += p // self.batch_size if self.drop_last else -(-p // self.batch_size)
        return total // self.num_replicas if self.num_replicas > 1 else total
