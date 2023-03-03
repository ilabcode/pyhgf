# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, Optional, Tuple, Union

from jax.interpreters.xla import DeviceArray

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
