# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

from jax import Array, jit

from pyhgf.math import sigmoid
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_binary_state_node(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    node_idx: int,
) -> Tuple[Array, ...]:
    r"""Get the new expected mean and precision of a binary state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the binary state node.

    Returns
    -------
    expected_precision :
        The precision of the value parent.
    expected_mean :
        The mean of the value parent.

    """
    # List the (unique) value parent of the value parent
    value_parent_idx = edges[node_idx].value_parents[0]  # type: ignore

    # Estimate the new expected mean of the value parent and apply the sigmoid transform
    expected_mean = attributes[value_parent_idx]["expected_mean"]
    expected_mean = sigmoid(expected_mean)

    # Estimate the new expected precision of the value parent
    expected_precision = 1 / (expected_mean * (1 - expected_mean))

    return expected_precision, expected_mean
