# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

from pyhgf.model import Network
from pyhgf.updates.prediction_error.inputs.generic import generic_input_prediction_error


class TestPredictionErrors(TestCase):
    def test_generic_input(self):
        """Test the generic input nodes"""

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        network = Network().add_nodes(kind="generic-input").add_nodes(value_children=0)

        attributes, (_, edges), _ = network.get_network()

        attributes = generic_input_prediction_error(
            attributes=attributes,
            time_step=1.0,
            edges=edges,
            node_idx=0,
            value=10.0,
            observed=True,
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
