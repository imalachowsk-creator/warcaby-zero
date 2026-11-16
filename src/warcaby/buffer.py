"""Replay buffer: the positions the network learns from.

Each stored position has three parts:

* the **observation** (what the network sees),
* the **policy target** - how MCTS split its simulations among the moves,
* the **value target** - how the game really ended for the player to move.

The buffer keeps the most recent positions only. Old games were played by a
weaker network, so their move choices and results are less trustworthy; a
sliding window forgets them while still mixing several iterations of data,
which keeps the training batches varied.

To stay small in memory and on disk, observations are stored as bytes (values
0..1 scaled to 0..255) and the policy target as a short list of
``(action index, probability)`` pairs instead of a mostly empty vector.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

#: at most this many moves are kept per position (the most visited ones)
MAX_POLICY_ENTRIES = 48

FIELDS = ("observations", "policy_indices", "policy_probs", "values")


def empty_samples(observation_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((0, *observation_shape), dtype=np.uint8),
        "policy_indices": np.zeros((0, MAX_POLICY_ENTRIES), dtype=np.int16),
        "policy_probs": np.zeros((0, MAX_POLICY_ENTRIES), dtype=np.float32),
        "values": np.zeros(0, dtype=np.float32),
    }


def concatenate_samples(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.concatenate([part[name] for part in parts]) for name in FIELDS}


def pack_observation(observation: np.ndarray) -> np.ndarray:
    return np.round(observation * 255.0).astype(np.uint8)


def pack_policy(policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A dense policy vector as padded ``(indices, probabilities)`` rows."""
    indices = np.flatnonzero(policy)
    if len(indices) > MAX_POLICY_ENTRIES:
        indices = indices[np.argsort(policy[indices])[-MAX_POLICY_ENTRIES:]]
    probs = policy[indices] / policy[indices].sum()
    packed_indices = np.zeros(MAX_POLICY_ENTRIES, dtype=np.int16)
    packed_probs = np.zeros(MAX_POLICY_ENTRIES, dtype=np.float32)
    packed_indices[: len(indices)] = indices
    packed_probs[: len(indices)] = probs
    return packed_indices, packed_probs


class ReplayBuffer:
    """A fixed-size window over the most recent positions."""

    def __init__(self, capacity: int, observation_shape: tuple[int, int, int]) -> None:
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self._data = {
            "observations": np.zeros((capacity, *self.observation_shape), dtype=np.uint8),
            "policy_indices": np.zeros((capacity, MAX_POLICY_ENTRIES), dtype=np.int16),
            "policy_probs": np.zeros((capacity, MAX_POLICY_ENTRIES), dtype=np.float32),
            "values": np.zeros(capacity, dtype=np.float32),
        }
        self._next = 0
        self._size = 0
        #: positions ever added, including those already forgotten
        self.total_added = 0

    def __len__(self) -> int:
        return self._size

    def add(self, samples: dict[str, np.ndarray]) -> None:
        count = len(samples["values"])
        if count > self.capacity:
            samples = {name: samples[name][-self.capacity :] for name in FIELDS}
            self.total_added += count - self.capacity
            count = self.capacity
        positions = (self._next + np.arange(count)) % self.capacity
        for name in FIELDS:
            self._data[name][positions] = samples[name]
        self._next = (self._next + count) % self.capacity
        self._size = min(self._size + count, self.capacity)
        self.total_added += count

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        chosen = rng.integers(0, self._size, size=batch_size)
        return {name: self._data[name][chosen] for name in FIELDS}

    def contents(self) -> dict[str, np.ndarray]:
        """All stored positions, oldest first."""
        if self._size < self.capacity:
            order = np.arange(self._size)
        else:
            order = (self._next + np.arange(self.capacity)) % self.capacity
        return {name: self._data[name][order] for name in FIELDS}

    def save(self, path: Path | str) -> None:
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, total_added=self.total_added, **self.contents())
        os.replace(temporary, path)

    def load(self, path: Path | str) -> None:
        with np.load(path) as stored:
            self._next = self._size = 0
            self.add({name: stored[name] for name in FIELDS})
            self.total_added = int(stored["total_added"])
