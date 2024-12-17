# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
    continuous_node_posterior_update_unbounded,
)
from pyhgf.updates.posterior.continuous.continuous_node_posterior_update_unbounded import (
    b,
    delta,
    mu_l,
    mu_l1,
    mu_l2,
    omega,
    pi_l,
    pi_l1,
    pi_l2,
    s,
)


def test_continuous_posterior_updates():

    network = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .add_nodes(volatility_children=2)
    )

    # Standard HGF updates -------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    attributes, edges, _ = network.get_network()
    _ = continuous_node_posterior_update(attributes=attributes, node_idx=2, edges=edges)

    # eHGF updates ---------------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    _ = continuous_node_posterior_update_ehgf(
        attributes=attributes, node_idx=2, edges=edges
    )

    # unbounded updates ----------------------------------------------------------------
    # ----------------------------------------------------------------------------------
    _ = continuous_node_posterior_update_unbounded(
        attributes=attributes, node_idx=2, edges=edges
    )


def test_unbounded_hgf_equations():

    alpha = 1.0
    beta = 5.0
    gamma = 4.0

    assert jnp.isclose(omega(alpha, gamma), 0.98201376)
    assert jnp.isclose(delta(alpha, beta, gamma), -0.9100689)

    assert b(1.0, 1.0, 1.0, 1.0, 1.0) == 0.25
    assert s(1.0, 1.0, 1.0) == 0.5

    assert jnp.isclose(pi_l1(alpha, gamma), 0.5088314)
    assert jnp.isclose(pi_l2(alpha, beta), 0.82389593)
    assert jnp.isclose(pi_l(alpha, beta, gamma), 0.51449823)

    assert jnp.isclose(mu_l1(alpha, beta, gamma), 3.1218112)
    assert jnp.isclose(mu_l2(alpha, beta, gamma), 2.9723248)
    assert jnp.isclose(mu_l(alpha, beta, gamma), 3.1191223)
