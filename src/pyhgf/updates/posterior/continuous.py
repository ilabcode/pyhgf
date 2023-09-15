# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


def continuous_node_update_mean_value_parent(
    attributes, edges, time_step, value_parent_idx, pi_value_parent, psi
) -> Array:
    # list the value and volatility parents
    value_parent_value_parents_idxs = edges[value_parent_idx].value_parents

    # Compute new muhat
    driftrate = attributes[value_parent_idx]["rho"]

    # Look at the (optional) va_pa's value parents
    # and update drift rate accordingly
    if value_parent_value_parents_idxs is not None:
        for va_pa_va_pa in value_parent_value_parents_idxs:
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


def continuous_node_update_precision_value_parent(
    attributes, edges, time_step, value_parent_idx
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
def continuous_node_update_value_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    value_parent_idx: int,
    psi: ArrayLike,
) -> Tuple[Array, ...]:
    pi_value_parent, nu_value_parent = continuous_node_update_precision_value_parent(
        attributes, edges, time_step, value_parent_idx
    )
    mu_value_parent = continuous_node_update_mean_value_parent(
        attributes, edges, time_step, value_parent_idx, pi_value_parent, psi
    )

    return pi_value_parent, mu_value_parent, nu_value_parent
