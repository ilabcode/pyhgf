# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node_unbounded(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
    precision_l1: float,
    precision_l2: float,
) -> float:
    """Posterior update of mean using ubounded update."""
    volatility_child_idx = edges[node_idx].volatility_children[0]
    volatility_coupling = attributes[node_idx]["volatility_coupling_children"][0]
    gamma = attributes[node_idx]["expected_mean"]
    phi = jnp.log(
        (1 / attributes[volatility_child_idx]["precision"]) * (2 + jnp.sqrt(3))
    )

    # first approximation ------------------------------------------------------
    delta_l1 = (
        (
            (1 / attributes[volatility_child_idx]["precision"])
            + (
                attributes[volatility_child_idx]["mean"]
                - attributes[volatility_child_idx]["expected_mean"] ** 2
            )
        )
        / (
            (1 / attributes[volatility_child_idx]["expected_precision"])
            + jnp.exp(
                volatility_coupling * phi
                + attributes[volatility_child_idx]["tonic_volatility"]
            )
        )
    ) - 1
    mean_l1 = (
        attributes[node_idx]["expected_mean"]
        + (
            (volatility_coupling * attributes[node_idx]["tonic_volatility"])
            / (2 * precision_l1)
        )
        * delta_l1
    )

    # second approximation -----------------------------------------------------
    omega_phi = jnp.exp(
        volatility_coupling * phi + attributes[node_idx]["tonic_volatility"]
    ) / (
        (1 / attributes[volatility_child_idx]["precision"])
        + jnp.exp(volatility_coupling * phi + attributes[node_idx]["tonic_volatility"])
    )
    delta_phi = (
        (1 / attributes[volatility_child_idx]["precision"])
        + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        )
        ** 2
    ) / (
        (1 / attributes[volatility_child_idx]["expected_precision"])
        + jnp.exp(
            volatility_coupling * phi
            + attributes[volatility_child_idx]["tonic_volatility"]
        )
    ) - 1

    mu_phi = ((2 * precision_l2 - 1) * phi + attributes[node_idx]["expected_mean"]) / (
        2 * precision_l2
    )

    mean_l2 = (
        mu_phi + (volatility_coupling * omega_phi) / (2 * precision_l2) * delta_phi
    )

    # weigthed interpolation
    theta_l = jnp.sqrt(
        1.2
        * (
            (1 / attributes[volatility_child_idx]["precision"])
            + (
                attributes[volatility_child_idx]["mean"]
                - attributes[volatility_child_idx]["expected_mean"]
            )
            ** 2
        )
        / ((1 / attributes[volatility_child_idx]["expected_precision"]) * precision_l1)
    )
    phi_l = 8.0
    theta_r = 0.0
    phi_r = 1.0
    mean = (1 - b(gamma, theta_l, phi_l, theta_r, phi_r)) * mean_l1 + b(
        gamma, theta_l, phi_l, theta_r, phi_r
    ) * mean_l2

    return mean


def s(x, theta, phi):
    return 1 / (1 + jnp.exp(-phi * (x - theta)))


def b(x, theta_l, phi_l, theta_r, phi_r):
    return s(x, theta_l, phi_l) - (1 - s(x, theta_r, phi_r))
