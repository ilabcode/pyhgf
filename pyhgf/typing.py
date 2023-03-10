# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, NamedTuple, Optional, Tuple, Union

from jax.interpreters.xla import DeviceArray


class Indexes(NamedTuple):
    """Indexes to a node's value and volatility parents."""

    value_parents: Optional[Tuple]
    volatility_parents: Optional[Tuple]


ParametersType = Dict[
    str, Optional[Union[DeviceArray, float, Tuple[DeviceArray, float]]]
]
NodeType = Tuple[
    ParametersType,
    Optional[Tuple],
    Optional[Tuple],
]
ParentsType = Optional[Tuple[NodeType]]

NodeStructure = Dict[str, NodeType]
