# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.continuous import continuous_input_update, continuous_node_update
from pyhgf.structure import beliefs_propagation
from pyhgf.typing import Indexes


class TestStructure(TestCase):
    def test_beliefs_propagation(self):
        """Test the loop_inputs function"""

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": None,
            "psis": None,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
        }
        node_parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": (1.0,),
            "mu": 1.0,
            "nu": 1.0,
            "psis": (1.0,),
            "omega": 1.0,
            "rho": 1.0,
        }

        node_structure = (
            Indexes((1,), None),
            Indexes(None, (2,)),
            Indexes(None, None),
        )
        parameters_structure = (
            input_node_parameters,
            node_parameters,
            node_parameters,
        )

        # create update sequence
        sequence1 = 0, continuous_input_update
        sequence2 = 1, continuous_node_update
        update_sequence = (sequence1, sequence2)

        # one batch of new observations with time step
        data = jnp.array([0.2, 1.0])

        # apply sequence
        beliefs_propagation(
            parameters_structure=parameters_structure,
            data=data,
            update_sequence=update_sequence,
            node_structure=node_structure,
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
