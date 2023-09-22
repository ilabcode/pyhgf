# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

from jax import Array, jit

from pyhgf.math import sigmoid
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def predict_input_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    r"""Prediction step for the value parent of a binary input node.

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
    value_parent_value_parent_idxs = edges[value_parent_idx].value_parents[0]

    # Get the drift rate from the value parent of the value parent
    driftrate = attributes[value_parent_value_parent_idxs]["rho"]

    # Look at the (optional) value parent's value parents
    # and update the drift rate accordingly
    if edges[value_parent_value_parent_idxs].value_parents is not None:
        for (
            value_parent_value_parent_value_parent_idx,
            psi_parent_parent,
        ) in zip(
            edges[value_parent_value_parent_idxs].value_parents,
            attributes[value_parent_value_parent_idxs]["psis_parents"],
        ):
            driftrate += (
                psi_parent_parent
                * attributes[value_parent_value_parent_value_parent_idx]["mu"]
            )

    # Estimate the new expected mean of the value parent and apply the sigmoid transform
    muhat_value_parent = (
        attributes[value_parent_value_parent_idxs]["mu"] + time_step * driftrate
    )
    muhat_value_parent = sigmoid(muhat_value_parent)

    # Estimate the new expected precision of the value parent
    pihat_value_parent = 1 / (muhat_value_parent * (1 - muhat_value_parent))

    return pihat_value_parent, muhat_value_parent
