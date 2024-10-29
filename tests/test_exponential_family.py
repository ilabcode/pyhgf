# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np
from rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork


def test_1d_gaussain():

    timeseries = load_data("continuous")

    # Rust -----------------------------------------------------------------------------
    rs_network = RsNetwork()
    rs_network.add_nodes(kind="exponential-state")
    rs_network.set_update_sequence()
    rs_network.input_data(timeseries)

    # Python ---------------------------------------------------------------------------
    py_network = PyNetwork().add_nodes(kind="exponential-state")
    py_network.input_data(timeseries)

    # Ensure identical results
    assert np.isclose(
        py_network.node_trajectories[0]["xis"], rs_network.node_trajectories[0]["xis"]
    ).all()
    assert np.isclose(
        py_network.node_trajectories[0]["mean"], rs_network.node_trajectories[0]["mean"]
    ).all()
    assert np.isclose(
        py_network.node_trajectories[0]["nus"], rs_network.node_trajectories[0]["nus"]
    ).all()
