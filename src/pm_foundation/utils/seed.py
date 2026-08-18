"""Deterministic seeding across Python, NumPy, and PyTorch."""

from __future__ import annotations


def seed_everything(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed all RNGs for reproducibility.

    Args:
        seed: The base seed.
        deterministic: If True, request deterministic algorithms where available.

    Returns:
        The seed used.
    """
    raise NotImplementedError("M0: implement RNG seeding (random/numpy/torch).")
