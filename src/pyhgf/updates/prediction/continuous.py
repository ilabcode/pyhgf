# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"), static_argnums=(2, 3))
def predict_mean(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
) -> Array:
    r"""Expected value for the mean of a probabilistic node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    expected_mean :
        The expected value for the mean of the value parent (:math:`\\hat{\\mu}`).

    """
    # List the node's value parents
    value_parents_idxs = edges[node_idx].value_parents

    # Get the drift rate from the node
    driftrate = attributes[node_idx]["tonic_drift"]

    # Look at the (optional) value parents for this node
    # and update the drift rate accordingly
    if value_parents_idxs is not None:
        for value_parent_idx, psi in zip(
            value_parents_idxs,
            attributes[node_idx]["value_coupling_parents"],
        ):
            driftrate += psi * attributes[value_parent_idx]["mean"]

    # New expected mean from the previous value
    expected_mean = attributes[node_idx]["mean"]

    # Take the drift into account
    expected_mean += time_step * driftrate

    # Add quatities that come from the autoregressive process if not zero
    expected_mean += (
        time_step
        * attributes[node_idx]["autoregressive_coefficient"]
        * (
            attributes[node_idx]["autoregressive_intercept"]
            - attributes[node_idx]["mean"]
        )
    )

    return expected_mean


@partial(jit, static_argnames=("edges", "node_idx"), static_argnums=(2, 3))
def predict_precision(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges
) -> Array:
    r"""Expected value for the precision of the value parent.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that will be updated.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    expected_precision :
        The expected value for the mean of the value parent (:math:`\\hat{\\pi}`).

    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the log volatility from the node
    logvol = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents
    # and update the log volatility accordingly
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            logvol += volatility_coupling * attributes[volatility_parents_idx]["mean"]

    # Estimate new nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # Estimate the new expected precision for the node
    expected_precision = 1 / (1 / attributes[node_idx]["precision"] + new_nu)

    return expected_precision
