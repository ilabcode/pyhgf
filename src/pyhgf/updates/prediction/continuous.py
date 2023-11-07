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
    r"""Compute the expected mean of a continuous state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    time_step :
        The time interval between the previous time point and the current time point.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_mean :
        The new expected mean of the value parent.

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


@partial(jit, static_argnames=("edges", "node_idx"))
def predict_precision(
    attributes: Dict, edges: Edges, time_step: float, node_idx: int
) -> Array:
    r"""Compute the expected precision of a continuous state node node.

    The expected precision at time :math:`k` for a state node :math:`a` is given by:

    .. math::

        \hat{\pi}_a^{(k)} = \frac{1}{\frac{1}{\pi_a^{(k-1)}} + \Omega_a^{(k)}}

    where :math:`\Omega_a^{(k)}` is the *total predicted volatility*. This term is the
    sum of the tonic (endogenous) and phasic (exogenous) volatility, such as:

    .. math::

        \Omega_a^{(k)} = t^{(k)} \\
        \exp{ \left( \omega_a + \sum_{j=1}^{N_{vopa}} \kappa_j \mu_a^{(k-1)} \right) }


    with :math:`\kappa_j` the volatility coupling strength with the volatility parent
    :math:`j`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    time_step :
        The time interval between the previous time point and the current time point.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    expected_precision :
        The new expected precision of the value parent.
    predicted volatility :
        The predicted volatility :math:`\Omega_a^{(k)}`. This value is stored in the
        node for latter use in the prediction-error steps.

    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling * attributes[volatility_parents_idx]["mean"]
            )

    # compute the predicted_volatility from the total volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Estimate the new expected precision for the node
    expected_precision = 1 / (
        (1 / attributes[node_idx]["precision"]) + predicted_volatility
    )

    return expected_precision, predicted_volatility
