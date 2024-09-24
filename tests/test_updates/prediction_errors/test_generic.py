# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from pyhgf.model import Network
from pyhgf.updates.prediction_error.generic import generic_state_prediction_error


def test_generic_input():
    """Test the generic input nodes"""

    ###############################################
    # one value parent with one volatility parent #
    ###############################################
    network = Network().add_nodes(kind="generic-state").add_nodes(value_children=0)

    attributes, edges, _ = network.get_network()

    attributes = generic_state_prediction_error(
        attributes=attributes,
        edges=edges,
        node_idx=0,
        value=10.0,
        observed=True,
    )
