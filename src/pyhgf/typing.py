# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, NamedTuple, Optional, Tuple


class AdjacencyLists(NamedTuple):
    """Indexes to a node's value and volatility parents."""

    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]
    value_children: Optional[Tuple]
    volatility_children: Optional[Tuple]


class Inputs(NamedTuple):
    """Input nodes type and index."""

    idx: Tuple[int, ...]
    kind: Tuple[int, ...]


# the nodes' attributes
Attributes = Dict[int, Dict]

# the network edges
Edges = Tuple[AdjacencyLists, ...]

# the network structure (the edges and the inputs info)
Structure = Tuple[Inputs, Edges]

# the update sequence
UpdateSequence = Tuple[Tuple[int, Callable], ...]

# a fully defined network
Network = Tuple[Attributes, Structure, UpdateSequence]

# encoding input types using intergers
input_types = {"continuous": 0, "binary": 1, "categorical": 2}
