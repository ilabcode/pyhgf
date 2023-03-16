# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Tuple


class Indexes(NamedTuple):
    """Indexes to a node's value and volatility parents."""

    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]


class InputIndexes(NamedTuple):
    """Input node type and index."""

    idx: int
    kind: str


NodeStructure = Tuple[Indexes, ...]

UpdateSequence = Tuple[Tuple[int, Callable], ...]
