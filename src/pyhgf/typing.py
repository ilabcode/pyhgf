# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, NamedTuple, Optional, Tuple, Type


class Indexes(NamedTuple):
    """Indexes to a node's value and volatility parents."""

    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]
    value_children: Optional[Tuple]
    volatility_children: Optional[Tuple]


class InputIndexes(NamedTuple):
    """Input nodes type and index."""

    idx: Tuple[int, ...]
    kind: Tuple[str, ...]


Edges = Tuple[Indexes, ...]

UpdateSequence = Tuple[Tuple[int, Callable], ...]


class DirichletNode(NamedTuple):
    """Static arguments of a Dirichlet node."""

    base_network: Type
    likelihood_fn: Callable
    cluster_input_idxs: Tuple
