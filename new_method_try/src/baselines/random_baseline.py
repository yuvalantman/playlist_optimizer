"""Random-order baseline: shuffle the interior, keep fixed endpoints."""

import numpy as np


class RandomBaseline:
    """Place all non-endpoint tracks in a uniformly random order."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def generate(
        self,
        number_of_tracks: int,
        start_index: int,
        end_index: int,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Return a random exact-once playlist with fixed start and end."""
        if number_of_tracks < 2:
            raise ValueError("number_of_tracks must be at least 2.")
        if start_index == end_index:
            raise ValueError("start_index and end_index must differ.")
        generator = rng if rng is not None else np.random.default_rng(self.seed)
        interior = [
            track
            for track in range(number_of_tracks)
            if track not in (start_index, end_index)
        ]
        generator.shuffle(interior)
        return [start_index, *interior, end_index]
