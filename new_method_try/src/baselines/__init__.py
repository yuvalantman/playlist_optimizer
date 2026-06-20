"""Baseline playlist constructors for the ablation ladder.

Each baseline exposes a ``generate(...) -> list[int]`` method returning a complete
exact-once playlist with the fixed start at position 0 and the fixed end at the last
position. Baselines deliberately optimize different (or no) objectives so that the
ablation isolates the contribution of each algorithmic ingredient.

Members:
- RandomBaseline            (floor; optimizes nothing)
- TransitionGreedyBaseline  (optimizes C_trans only)
- ArcAssignmentBaseline     (optimal C_arc assignment; arc_rmse lower bound)
- ForwardBeamBaseline       (arc + transition, one direction)
- MMBeamBaseline            (arc + transition, bidirectional, non-recursive, commits all)
- FlexerInterpolationBaseline (start/end acoustic interpolation, Flexer 2008)
"""

from src.baselines.arc_assignment import ArcAssignmentBaseline
from src.baselines.flexer_interp import FlexerInterpolationBaseline
from src.baselines.forward_beam import ForwardBeamBaseline
from src.baselines.mm_beam import MMBeamBaseline
from src.baselines.random_baseline import RandomBaseline
from src.baselines.transition_greedy import TransitionGreedyBaseline

__all__ = [
    "ArcAssignmentBaseline",
    "FlexerInterpolationBaseline",
    "ForwardBeamBaseline",
    "MMBeamBaseline",
    "RandomBaseline",
    "TransitionGreedyBaseline",
]
