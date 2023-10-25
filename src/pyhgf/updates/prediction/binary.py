# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array

from pyhgf.math import sigmoid
from pyhgf.typing import Edges


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
    value_parent_idx :
        Pointer to the value parent node.

    Returns
    -------
    pi_value_parent :
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.
    """
    # Get the expected mean from the value parent and apply the sigmoid transform
    expected_mean = jnp.where(
        edges["value_parents"][node_idx], attributes["expected_mean"], 0.0
    ).sum()
    expected_mean = sigmoid(expected_mean)

    # Estimate the new expected precision of the value parent
    expected_precision = 1 / (expected_mean * (1 - expected_mean))

    return expected_precision, expected_mean
