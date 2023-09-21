# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_mean(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    node_idx: int,
) -> Array:
    r"""Expected value for the mean of a probabilistic node.

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
        Pointer to the node that will be updated.

    Returns
    -------
    muhat :
        The expected value for the mean of the value parent (:math:`\\hat{\\mu}`).

    """
    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = attributes[node_idx]["rho"]

    # Look at the (optional) value parents for this node
    # and update the drift rate accordingly
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[value_parent_idx]["mu"]

    # Compute the new expected mean this node
    muhat = attributes[node_idx]["mu"] + time_step * driftrate

    return muhat


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision(
    attributes: Dict, edges: Edges, time_step: float, node_idx: int
) -> Array:
    r"""Expected value for the precision of the value parent.

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
        Pointer to the node that will be updated.

    Returns
    -------
    pihat :
        The expected value for the mean of the value parent (:math:`\\hat{\\pi}`).

    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the log volatility from the node
    logvol = attributes[node_idx]["omega"]

    # Look at the (optional) volatility parents
    # and update the log volatility accordingly
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, k in zip(
            volatility_parents_idxs,
            attributes[node_idx]["kappas_parents"],
        ):
            logvol += k * attributes[volatility_parents_idx]["mu"]

    # Estimate new nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # Estimate the new expected precision for the node
    pihat = 1 / (1 / attributes[node_idx]["pi"] + new_nu)

    return pihat
