# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    r"""Expected value for the mean of the value parent.

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
        Pointer to the node that will be updated.

    Returns
    -------
    muhat_value_parent :
        The expected value for the mean of the value parent (:math:`\\hat{\\mu}`).

    """
    # list the value and volatility parents
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # drift rate of the parent node
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) valu parents of the value parents and update drift rate
    if value_parent_value_parents_idxs is not None:
        for va_pa_va_pa, psi in zip(
            value_parent_value_parents_idxs,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[va_pa_va_pa]["mu"]

    # Compute new expected value
    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate

    return muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_precision_value_parent(
    attributes: Dict, edges: Edges, time_step: float, value_parent_idx: int
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
    value_parent_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    pihat_value_parent :
        The expected value for the mean of the value parent (:math:`\\hat{\\pi}`).

    """
    # list the value and volatility parents
    value_parent_volatility_parents_idxs = edges[value_parent_idx].volatility_parents

    # Compute new value for nu and pihat
    logvol = attributes[value_parent_idx]["omega"]

    # Look at the (optional) va_pa's volatility parents
    # and update logvol accordingly
    if value_parent_volatility_parents_idxs is not None:
        for value_parent_volatility_parents_idx, k in zip(
            value_parent_volatility_parents_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parents_idx]["mu"]

    # Estimate new_nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    pihat_value_parent = 1 / (1 / attributes[value_parent_idx]["pi"] + new_nu)

    return pihat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Prediction step for the value parent(s) of a continuous node.

    Updating the posterior distribution of the value parent is a two-step process:
    1. Update the posterior precision using
    :py:func:`pyhgf.updates.prediction.continuous.prediction_precision_value_parent`.
    2. Update the posterior mean value using
    :py:func:`pyhgf.updates.prediction.continuous.prediction_mean_value_parent`.

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

    pi_value_parent = prediction_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    mu_value_parent = prediction_mean_value_parent(
        attributes, edges, time_step, value_parent_idx
    )

    return pi_value_parent, mu_value_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_precision_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    # list the volatility parents
    volatility_parent_volatility_parents_idx = edges[
        volatility_parent_idx
    ].volatility_parents

    # Compute new value for nu and pihat
    logvol = attributes[volatility_parent_idx]["omega"]

    # Look at the (optional) vo_pa's volatility parents
    # and update logvol accordingly
    if volatility_parent_volatility_parents_idx is not None:
        for vo_pa_vo_pa, k in zip(
            volatility_parent_volatility_parents_idx,
            attributes[volatility_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[vo_pa_vo_pa]["mu"]

    # Estimate new_nu
    new_nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(new_nu > 1e-128, new_nu, jnp.nan)
    pihat_volatility_parent = 1 / (1 / attributes[volatility_parent_idx]["pi"] + new_nu)

    return pihat_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_mean_volatility_parent(
    attributes, edges, time_step, volatility_parent_idx
) -> Array:
    # list the value parents
    volatility_parent_value_parents_idx = edges[volatility_parent_idx].value_parents

    # drift rate of the GRW
    driftrate = attributes[volatility_parent_idx]["rho"]

    # Look at the (optional) va_pa's value parents
    # and update drift rate accordingly
    if volatility_parent_value_parents_idx is not None:
        for vo_pa_va_pa, psi in zip(
            volatility_parent_value_parents_idx,
            attributes[volatility_parent_idx]["psi_parents"],
        ):
            driftrate += psi * attributes[vo_pa_va_pa]["mu"]

    muhat_volatility_parent = (
        attributes[volatility_parent_idx]["mu"] + time_step * driftrate
    )

    return muhat_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_volatility_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    volatility_parent_idx: int,
) -> Tuple[Array, ...]:
    r"""Prediction step for the volatility parent(s) of a continuous node.

    Updating the posterior distribution of the volatility parent is a two-step process:
    1. Update the posterior precision using
    :py:fun:`update_precision_volatility_parent`.
    2. Update the posterior mean value using
    :py:fun:`update_mean_volatility_parent`.

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
    volatility_parent_idx :
        Pointer to the value parent node.

    Returns
    -------
    pi_value_parent :
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.
    nu_value_parent :

    """
    pi_volatility_parent = prediction_precision_volatility_parent(
        attributes, edges, time_step, volatility_parent_idx
    )
    mu_volatility_parent = prediction_mean_volatility_parent(
        attributes, edges, time_step, volatility_parent_idx
    )

    return pi_volatility_parent, mu_volatility_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_input_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    muhat_value_parent = prediction_input_mean_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    pihat_value_parent = prediction_input_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )

    return pihat_value_parent, muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_input_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    # list value parents
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # Compute new muhat
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) va_pa's value parents
    # and update drift rate accordingly
    if value_parent_value_parents_idxs is not None:
        for value_parent_value_parents_idx in value_parent_value_parents_idxs:
            driftrate += (
                attributes[value_parent_idx]["psis_parents"][0]
                * attributes[value_parent_value_parents_idx]["mu"]
            )

    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate

    return muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_input_precision_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    # list volatility parents
    value_parent_volatility_parents_idxs = edges[value_parent_idx].volatility_parents

    # Compute new value for nu and pihat
    logvol = attributes[value_parent_idx]["omega"]

    # Look at the (optional) va_pa's volatility parents
    # and update logvol accordingly
    if value_parent_volatility_parents_idxs is not None:
        for value_parent_volatility_parents_idx, k in zip(
            value_parent_volatility_parents_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parents_idx]["mu"]

    # Estimate new_nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)
    pihat_value_parent = 1 / (1 / attributes[value_parent_idx]["pi"] + new_nu)

    return pihat_value_parent
