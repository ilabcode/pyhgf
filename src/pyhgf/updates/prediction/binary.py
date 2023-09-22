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
    value_parent_idx :
        Pointer to the value parent node.

    Returns
    -------
    pi_value_parent :
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.
    """
    # List the (unique) value parent of the value parent
    value_parent_idx = edges[node_idx].value_parents[0]

    # Get the drift rate from the value parent of the value parent
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) value parent's value parents
    # and update the drift rate accordingly
    if edges[value_parent_idx].value_parents is not None:
        for (
            value_parent_value_parent_idx,
            psi_parent,
        ) in zip(
            edges[value_parent_idx].value_parents,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi_parent * attributes[value_parent_value_parent_idx]["mu"]

    # Estimate the new expected mean of the value parent and apply the sigmoid transform
    muhat = attributes[value_parent_idx]["mu"] + time_step * driftrate
    muhat = sigmoid(muhat)

    # Estimate the new expected precision of the value parent
    pihat = 1 / (muhat * (1 - muhat))

    return pihat, muhat
