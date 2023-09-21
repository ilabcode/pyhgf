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
    # List the value and volatility parents of the value parent
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # Get the drift rate of the value parent
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) value parents of the value parent
    # and update the drift rate accordingly
    if value_parent_value_parents_idxs is not None:
        for value_parent_value_parent_idx, psi in zip(
            value_parent_value_parents_idxs,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[value_parent_value_parent_idx]["mu"]

    # Compute the new expected mean for the value parent
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
    # List the value parent's volatility parents
    value_parent_volatility_parents_idxs = edges[value_parent_idx].volatility_parents

    # Get the log volatility from the value parent
    logvol = attributes[value_parent_idx]["omega"]

    # Look at the (optional) value parent's volatility parents
    # and update the log volatility accordingly
    if value_parent_volatility_parents_idxs is not None:
        for value_parent_volatility_parents_idx, k in zip(
            value_parent_volatility_parents_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parents_idx]["mu"]

    # Estimate new nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # Estimate the new expected precision for the value parent
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
    #. Update the posterior precision using
    :py:func:`pyhgf.updates.prediction.continuous.prediction_precision_value_parent`.
    #. Update the posterior mean using
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
    # Get the new estimation for the value parent's precision
    pi_value_parent = prediction_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    # Get the new estimation for the value parent's mean
    mu_value_parent = prediction_mean_value_parent(
        attributes, edges, time_step, value_parent_idx
    )

    return pi_value_parent, mu_value_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_precision_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    r"""Expected value for the precision of the volatility parent of a contiuous node.

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
        Pointer to the volatility parent node that will be updated.

    Returns
    -------
    pihat_volatility_parent :
        The new expected value for the mean of the volatility parent
        (:math:`\\hat{\\pi}`).

    """
    # List the volatility parent's volatility parents
    volatility_parent_volatility_parents_idx = edges[
        volatility_parent_idx
    ].volatility_parents

    # Get the log volatility from the volatility parent
    logvol = attributes[volatility_parent_idx]["omega"]

    # Look at the (optional) volatility parent's volatility parents
    # and update the log volatility accordingly
    if volatility_parent_volatility_parents_idx is not None:
        for vo_pa_vo_pa, k in zip(
            volatility_parent_volatility_parents_idx,
            attributes[volatility_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[vo_pa_vo_pa]["mu"]

    # Estimate new_nu
    new_nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(new_nu > 1e-128, new_nu, jnp.nan)

    # Estimate the new expected precision for the volatility parent
    pihat_volatility_parent = 1 / (1 / attributes[volatility_parent_idx]["pi"] + new_nu)

    return pihat_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_mean_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    r"""Expected value for the mean of the volatility parent of a contiuous node.

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
        Pointer to the volatility parent node that will be updated.

    Returns
    -------
    muhat_volatility_parent :
        The new expected value for the mean of the volatility parent
        (:math:`\\hat{\\mu}`).

    """
    # List the volatility parent's value parents
    volatility_parent_value_parents_idx = edges[volatility_parent_idx].value_parents

    # Get the drift rate from the volatility parent
    driftrate = attributes[volatility_parent_idx]["rho"]

    # Look at the (optional) volatility parent's value parents
    # and update the drift rate accordingly
    if volatility_parent_value_parents_idx is not None:
        for vo_pa_va_pa, psi in zip(
            volatility_parent_value_parents_idx,
            attributes[volatility_parent_idx]["psi_parents"],
        ):
            driftrate += psi * attributes[vo_pa_va_pa]["mu"]

    # Estimate the new expected mean for the volatility parent
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
    #. Update the posterior precision using
    :py:fun:`update_precision_volatility_parent`.
    #. Update the posterior mean value using
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
    pi_volatility_parent :
        The precision (:math:`\\pi`) of the volatility parent.
    mu_volatility_parent :
        The mean (:math:`\\mu`) of the volatility parent.

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
    """Prediction step for the value parent of a continuous input node.

    Updating the posterior distribution of the value parent of a continuous input node
    is a two-step process:
    #. Update the parent's expected mean using
    :py:func:`pyhgf.updates.prediction.continuous.prediction_input_mean_value_parent`.
    #. Update the parent's expected precision using
    :py:func:`pyhgf.updates.prediction.continuous.prediction_input_precision_value_parent`.

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
    pihat_value_parent :
        The precision (:math:`\\hat{\\pi}`) of the value parent.
    muhat_value_parent :
        The mean (:math:`\\hat{\\mu}`) of the value parent.

    """
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
    r"""Expected value for the mean of the input's value parent.

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
    # list the value parent's value parents
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # Get the value parent's log volatility
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) value parent's value parents
    # and update the drift rate accordingly
    if value_parent_value_parents_idxs is not None:
        for value_parent_value_parents_idx in value_parent_value_parents_idxs:
            driftrate += (
                attributes[value_parent_idx]["psis_parents"][0]
                * attributes[value_parent_value_parents_idx]["mu"]
            )

    # Estimate the new expected mean for the value parent
    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate

    return muhat_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_input_precision_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Array:
    r"""Expected value for the precision of the input's value parent.

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
    # List volatility parents
    value_parent_volatility_parents_idxs = edges[value_parent_idx].volatility_parents

    # Get the log volatility from the value parent
    logvol = attributes[value_parent_idx]["omega"]

    # Look at the (optional) value parent's volatility parents
    # and update the log volatility accordingly
    if value_parent_volatility_parents_idxs is not None:
        for value_parent_volatility_parents_idx, k in zip(
            value_parent_volatility_parents_idxs,
            attributes[value_parent_idx]["kappas_parents"],
        ):
            logvol += k * attributes[value_parent_volatility_parents_idx]["mu"]

    # Estimate new nu
    nu = time_step * jnp.exp(logvol)
    new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

    # Estimate the new expected precision for the value parent
    pihat_value_parent = 1 / (1 / attributes[value_parent_idx]["pi"] + new_nu)

    return pihat_value_parent
