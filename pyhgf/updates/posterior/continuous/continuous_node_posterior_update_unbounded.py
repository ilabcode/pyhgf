# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_posterior_update_unbounded(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a continuous node using an unbounded approximation.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_posterior_update_ehgf

    """
    # update the posterior mean and precision using the eHGF update step
    # we start with the mean update using the expected precision as an approximation
    posterior_precision = posterior_update_precision_continuous_node_unbounded(
        attributes=attributes,
        edges=edges,
        node_idx=node_idx,
    )
    attributes[node_idx]["precision"] = posterior_precision

    posterior_mean = posterior_update_mean_continuous_node_unbounded(
        attributes=attributes,
        edges=edges,
        node_idx=node_idx,
    )
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node_unbounded(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
) -> float:
    """Posterior update of mean using ubounded update."""
    volatility_child_idx = edges[node_idx].volatility_children[0]  # type: ignore
    # volatility_coupling = attributes[node_idx]["volatility_coupling_children"][0]
    gamma = attributes[node_idx]["expected_mean"]

    # previous child uncertainty
    alpha = 1 / attributes[volatility_child_idx]["expected_precision"]

    # posterior total uncertainty about the child
    beta = (
        1 / attributes[volatility_child_idx]["expected_precision"]
        + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        )
        ** 2
    )

    return mu_l(alpha, beta, gamma)


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node_unbounded(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
) -> float:
    """Posterior update of mean using ubounded update."""
    volatility_child_idx = edges[node_idx].volatility_children[0]  # type: ignore
    # volatility_coupling = attributes[node_idx]["volatility_coupling_children"][0]
    gamma = attributes[node_idx]["expected_mean"]

    # previous child uncertainty
    alpha = 1 / attributes[volatility_child_idx]["expected_precision"]

    # posterior total uncertainty about the child
    beta = (
        1 / attributes[volatility_child_idx]["expected_precision"]
        + (
            attributes[volatility_child_idx]["mean"]
            - attributes[volatility_child_idx]["expected_mean"]
        )
        ** 2
    )

    return pi_l(alpha, beta, gamma)


def s(x, theta, psi):
    return 1 / (1 + jnp.exp(-psi * (x - theta)))


def b(x, theta_l, phi_l, theta_r, phi_r):
    return s(x, theta_l, phi_l) * (1 - s(x, theta_r, phi_r))


def pi_l1(alpha, gamma):
    return 0.5 * omega(alpha, gamma) * (1 - omega(alpha, gamma)) + 0.5


def mu_l1(alpha, beta, gamma):
    return gamma + 0.5 / pi_l1(alpha, gamma) * omega(alpha, gamma) * delta(
        alpha, beta, gamma
    )


def omega(alpha, x):
    return jnp.exp(x) / (alpha + jnp.exp(x))


def delta(alpha, beta, x):
    return beta / (alpha + jnp.exp(x)) - 1


def phi(alpha):
    return jnp.log(alpha * (2 + jnp.sqrt(3)))


def pi_l2(alpha, beta):
    return -ddJ(phi(alpha), alpha, beta)


def dJ(x, alpha, beta, gamma):
    return 0.5 * omega(alpha, x) * delta(alpha, beta, x) - 0.5 * (x - gamma)


def ddJ(x, alpha, beta):
    return (
        -0.5
        * omega(alpha, x)
        * (omega(alpha, x) + (2 * omega(alpha, x) - 1) * delta(alpha, beta, x))
        - 0.5
    )


def mu_l2(alpha, beta, gamma):
    return phi(alpha) - dJ(phi(alpha), alpha, beta, gamma) / ddJ(
        phi(alpha), alpha, beta
    )


def mu_l(alpha, beta, gamma):
    return (1 - b(gamma, -jnp.sqrt(1.2 * 2 * beta / alpha), 8.0, 0.0, 1.0)) * mu_l1(
        alpha, beta, gamma
    ) + b(gamma, -jnp.sqrt(1.2 * 2 * beta / alpha), 8.0, 0.0, 1.0) * mu_l2(
        alpha, beta, gamma
    )


def pi_l(alpha, beta, gamma):
    return (1 - b(gamma, -jnp.sqrt(1.2 * 2 * beta / alpha), 8.0, 0.0, 1.0)) * pi_l1(
        alpha, gamma
    ) + b(gamma, -jnp.sqrt(1.2 * 2 * beta / alpha), 8.0, 0.0, 1.0) * pi_l2(alpha, beta)
