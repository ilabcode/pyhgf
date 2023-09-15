# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def update_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
    pi_value_parent: ArrayLike,
) -> Array:
    # list the value and volatility parents
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # Compute new muhat
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) valu parents of the value parents and update drift rate
    if value_parent_value_parents_idxs is not None:
        for va_pa_va_pa, psi in zip(
            value_parent_value_parents_idxs,
            attributes[value_parent_idx]["psis_parents"],
        ):
            driftrate += psi * attributes[va_pa_va_pa]["mu"]

    muhat_value_parent = attributes[value_parent_idx]["mu"] + time_step * driftrate

    # gather PE updates from other nodes if the parent has many children
    # this part corresponds to the sum of children required for the
    # multi-children situations
    pe_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        vape_child = attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]
        pihat_child = attributes[child_idx]["pihat"]
        pe_children += (psi_child * pihat_child * vape_child) / pi_value_parent

    mu_value_parent = muhat_value_parent + pe_children

    return mu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def update_precision_value_parent(
    attributes: Dict, edges: Edges, time_step: float, value_parent_idx: int
) -> Array:
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

    pihat_value_parent, nu_value_parent = [
        1 / (1 / attributes[value_parent_idx]["pi"] + new_nu),
        new_nu,
    ]

    # gather precision updates from other nodes if the parent has many
    # children - this part corresponds to the sum over children
    # required for the multi-children situations
    pi_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        pihat_child = attributes[child_idx]["pihat"]
        pi_children += psi_child**2 * pihat_child

    pi_value_parent = pihat_value_parent + pi_children

    return pi_value_parent, nu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def update_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Update the mean and precision of the value parent of a continuous node.

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
    nu_value_parent :

    """
    pi_value_parent, nu_value_parent = update_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    mu_value_parent = update_mean_value_parent(
        attributes, edges, time_step, value_parent_idx, pi_value_parent
    )

    return pi_value_parent, mu_value_parent, nu_value_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def update_volatility_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    volatility_parent_idx: int,
) -> Tuple[Array, ...]:
    """Update the mean and precision of the volatility parent of a continuous node.

    Updating the posterior distribution of the volatility parent is a two-step process:
    1. Update the posterior precision using
    :py:fun:`update_precision_volatility_parent`.
    2. Update the posterior mean value using
    :py:fun:`update_mean_volatility_parent`.

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
    pi_volatility_parent, nu_volatility_parent = update_precision_volatility_parent(
        attributes, edges, time_step, volatility_parent_idx
    )
    mu_volatility_parent = update_mean_volatility_parent(
        attributes, edges, time_step, volatility_parent_idx, pi_volatility_parent
    )

    return pi_volatility_parent, mu_volatility_parent, nu_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def update_precision_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    # list the value parents of the volatility parent
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

    pihat_volatility_parent, nu_volatility_parent = [
        1 / (1 / attributes[volatility_parent_idx]["pi"] + new_nu),
        new_nu,
    ]

    # gather volatility precisions from the child nodes
    children_volatility_precision = 0.0
    for child_idx, kappas_children in zip(
        edges[volatility_parent_idx].volatility_children,
        attributes[volatility_parent_idx]["kappas_children"],
    ):
        nu_children = attributes[child_idx]["nu"]
        pihat_children = attributes[child_idx]["pihat"]
        pi_children = attributes[child_idx]["pi"]
        vope_children = (
            1 / attributes[child_idx]["pi"]
            + (attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]) ** 2
        ) * attributes[child_idx]["pihat"] - 1

        children_volatility_precision += (
            0.5
            * (kappas_children * nu_children * pihat_children) ** 2
            * (1 + (1 - 1 / (nu_children * pi_children)) * vope_children)
        )

    pi_volatility_parent = pihat_volatility_parent + children_volatility_precision

    pi_volatility_parent = jnp.where(
        pi_volatility_parent <= 0, jnp.nan, pi_volatility_parent
    )

    return pi_volatility_parent, nu_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def update_mean_volatility_parent(
    attributes, edges, time_step, volatility_parent_idx, pi_volatility_parent: ArrayLike
) -> Array:
    # list the volatility parents of the volatility parent
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

    # gather volatility prediction errors from the child nodes
    children_volatility_prediction_error = 0.0
    for child_idx, kappas_children in zip(
        edges[volatility_parent_idx].volatility_children,
        attributes[volatility_parent_idx]["kappas_children"],
    ):
        nu_children = attributes[child_idx]["nu"]
        pihat_children = attributes[child_idx]["pihat"]
        vope_children = (
            1 / attributes[child_idx]["pi"]
            + (attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]) ** 2
        ) * attributes[child_idx]["pihat"] - 1
        children_volatility_prediction_error += (
            0.5
            * kappas_children
            * nu_children
            * pihat_children
            / pi_volatility_parent
            * vope_children
        )

    mu_volatility_parent = (
        muhat_volatility_parent + children_volatility_prediction_error
    )

    return mu_volatility_parent
