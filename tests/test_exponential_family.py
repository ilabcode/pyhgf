# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from rshgf import Network as RsNetwork

from pyhgf import load_data
from pyhgf.model import Network as PyNetwork


def test_1d_gaussain():

    timeseries = load_data("continuous")

    # Rust -----------------------------------------------------------------------------
    rs_network = RsNetwork()
    rs_network.add_nodes(kind="exponential-state")
    rs_network.add_nodes(kind="exponential-state")
    rs_network.inputs
    rs_network.edges

    rs_network.input_data([[0, 0], [1, 1]])
    rs_network.input_data([timeseries, timeseries])

    # Python ---------------------------------------------------------------------------
    py_network = PyNetwork().add_nodes(kind="continuous-state")
    py_network.attributes
