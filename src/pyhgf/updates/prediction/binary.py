# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.math import sgm
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    value_parent_value_parent_idxs = edges[value_parent_idx].value_parents

    # drift rate
    driftrate = attributes[value_parent_idx]["rho"]

    # look at value parent's value parents and update driftrate accordingly
    if value_parent_value_parent_idxs is not None:
        for value_parent_value_parent_idx, psi in zip(
            value_parent_value_parent_idxs,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[value_parent_value_parent_idx]["mu"]

    # compute new muhat
    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate
    return muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_precision_value_parent(
    attributes: Dict, edges: Edges, time_step: float, value_parent_idx: int
) -> Array:
    value_parent_volatility_parent_idxs = edges[value_parent_idx].volatility_parents

    # get log volatility
    logvol = attributes[value_parent_idx]["omega"]

    # look at the va_pa's volatility parents and update accordingly
    if value_parent_volatility_parent_idxs is not None:
        for value_parent_volatility_parent_idx, k in zip(
            value_parent_volatility_parent_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parent_idx]["mu"]

    # compute new_nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # compute new value for pihat
    pihat_value_parent = 1 / (1 / attributes[value_parent_idx]["pi"] + new_nu)

    return pihat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Prediction step for the value parent(s) of a binary node.

    Updating the posterior distribution of the value parent is a two-step process:
    1. Update the posterior precision using
    :py:fun:`continuous_node_update_precision_value_parent`.
    2. Update the posterior mean value using
    :py:fun:`continuous_node_update_mean_value_parent`.

    Parameters
    ----------
    attributes :
        The nodes' parameters.
    edges :
        The edges of the network as a tuple of :py:class:`pyhgf.typing.Indexes` with
        the same length as node number. For each node, the index list value and
        volatility parents.
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

    pihat_value_parent = prediction_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    muhat_value_parent = prediction_mean_value_parent(
        attributes, edges, time_step, value_parent_idx
    )

    return pihat_value_parent, muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_input_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Prediction step for the value parent(s) of a binary input node.

    Updating the posterior distribution of the value parent is a two-step process:
    1. Update the posterior precision using
    :py:fun:`continuous_node_update_precision_value_parent`.
    2. Update the posterior mean value using
    :py:fun:`continuous_node_update_mean_value_parent`.

    Parameters
    ----------
    attributes :
        The nodes' parameters.
    edges :
        The edges of the network as a tuple of :py:class:`pyhgf.typing.Indexes` with
        the same length as node number. For each node, the index list value and
        volatility parents.
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
    # list the (unique) value parents
    value_parent_value_parent_idxs = edges[value_parent_idx].value_parents[0]

    # 1. Compute new muhat_value_parent and pihat_value_parent
    # --------------------------------------------------------
    # 1.1 Compute new_muhat from continuous node parent (x2)
    # 1.1.1 get rho from the value parent of the binary node (x2)
    driftrate = attributes[value_parent_value_parent_idxs]["rho"]

    # # 1.1.2 Look at the (optional) value parent's value parents (x3)
    # # and update the drift rate accordingly
    if edges[value_parent_value_parent_idxs].value_parents is not None:
        for (
            value_parent_value_parent_value_parent_idx,
            psi_parent_parent,
        ) in zip(
            edges[value_parent_value_parent_idxs].value_parents,
            attributes[value_parent_value_parent_idxs]["psis_parents"],
        ):
            # For each x2's value parents (optional)
            driftrate += (
                psi_parent_parent
                * attributes[value_parent_value_parent_value_parent_idx]["mu"]
            )

    # 1.1.3 compute new_muhat
    muhat_value_parent = (
        attributes[value_parent_value_parent_idxs]["mu"] + time_step * driftrate
    )

    muhat_value_parent = sgm(muhat_value_parent)
    pihat_value_parent = 1 / (muhat_value_parent * (1 - muhat_value_parent))

    return pihat_value_parent, muhat_value_parent
