# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional, Tuple, Union

from jax.interpreters.xla import DeviceArray

ParametersType = Dict[
    str, Optional[Union[DeviceArray, float, Tuple[DeviceArray, float]]]
]
NodeType = Tuple[
    ParametersType,
    Optional[Tuple[Tuple[ParametersType, Optional[Tuple], Optional[Tuple]]]],
    Optional[Tuple[Tuple[ParametersType, Optional[Tuple], Optional[Tuple]]]],
]
ParentsType = Optional[Tuple[NodeType]]


def node_validation(node: Tuple):
    """ "Verify that the node structure is valid."""

    assert len(node) == 3
    assert isinstance(node[0], Dict)
    if node[1] is not None:
        assert isinstance(node[1], tuple)
    if node[2] is not None:
        assert isinstance(node[2], tuple)

    for n in [1, 2]:
        if node[n] is not None:
            assert isinstance(node[n], tuple)
            assert len(node[n]) > 0

            for i in range(len(node[n])):
                node_validation(node[n][i])
