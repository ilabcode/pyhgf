# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase
from pyhgf.continuous import continuous_input_update, continuous_node_update
from pyhgf.structure import apply_sequence
from pyhgf.typing import Indexes
import jax.numpy as jnp

class TestStructure(TestCase):

    def test_apply_sequence(self):
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

        # apply sequence
        new_parameters_structure = apply_sequence(
            node_structure=node_structure,
            parameters_structure=parameters_structure,
            update_sequence=update_sequence, 
            time_step=1.0,
            values=jnp.array([.2]),
            input_nodes_idx=jnp.array([0])
            )

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
