# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.model import Network
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
    continuous_node_posterior_update_unbounded,
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
