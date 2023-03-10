# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Tuple


class Indexes(NamedTuple):
    """Indexes to a node's value and volatility parents."""

    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]


NodeStructure = Tuple[Indexes, ...]

UpdateSequence = Tuple[Tuple[int, Callable], ...]
