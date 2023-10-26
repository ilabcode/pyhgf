# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict

import jax.numpy as jnp
from jax import Array

from pyhgf.typing import Edges


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
    # Get the drift rate from the node
    driftrate = attributes["tonic_drift"][node_idx]

    # Look at the (optional) value parents for this node
    # and update the drift rate accordingly
    mean_value_parents = jnp.where(
        edges["value_parents"][node_idx], attributes["mean"], 0.0
    )
    driftrate += (
        attributes["value_coupling_parents"][node_idx] * mean_value_parents
    ).sum()

    # New expected mean from the previous value
    expected_mean = attributes["mean"][node_idx]

    # Take the drift into account
    expected_mean += time_step * driftrate

    # Add quantities that come from the autoregressive process if not zero
    expected_mean += (
        time_step
        * attributes["autoregressive_coefficient"][node_idx]
        * (
            attributes["autoregressive_intercept"][node_idx]
            - attributes["mean"][node_idx]
        )
    )

    return expected_mean


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
    # Get the log volatility from the node
    logvol = attributes["tonic_volatility"][node_idx]

    # Look at the (optional) volatility parents
    # and update the log volatility accordingly
    mean_volatility_parents = jnp.where(
        edges["volatility_parents"][node_idx], attributes["mean"], 0.0
    )
    logvol += (
        attributes["volatility_coupling_parents"][node_idx] * mean_volatility_parents
    ).sum()

    # Estimate new nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # Estimate the new expected precision for the node
    expected_precision = 1 / (1 / attributes["precision"][node_idx] + new_nu)

    return expected_precision
