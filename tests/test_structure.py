# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

from pyhgf import load_data
from pyhgf.model import HGF
from pyhgf.structure import structure_as_dict, structure_validation
from pyhgf.continuous import continuous_input_update

class TestStructure(TestCase):

    def test_loop_inputs(self):
        """Test the loop_inputs function"""

        ############################################
        # one value parent - one volatility parent #
        ############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": 1.0,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
        }
        parameters = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "kappas": None,
            "mu": 1.0,
            "nu": 1.0,
            "psis": (1.0,),
            "omega": 1.0,
            "rho": 1.0,
        }
        volatility_parent_node = {
            "parameters": parameters,
            "value_parents": None,
            "volatility_parents": None
        }
        value_parent_node = {
            "parameters": parameters,
            "value_parents": None,
            "volatility_parents": None
        }
        input_node = {
            "parameters": input_node_parameters,
            "value_parents": (1,),
            "volatility_parents": (2,),
            }
        node_structure = {
            0: input_node, 
            1: value_parent_node,
            2: volatility_parent_node,
            }

        data = (5.0, 1.0)  # value, new_time_step

        sequence1 = 0, (1,), (2,), continuous_input_update
        update_sequence = (sequence1,)

        _, last = loop_inputs(
            update_sequence=update_sequence,
            node_structure=node_structure, 
            data=data,
            )
        # unwrap the new node structure

        assert last[0]["parameters"]["time_step"] == 1.0
        assert last[0]["parameters"]["value"] == 5.0
        assert jnp.isclose(last[0]["parameters"]["surprise"], 2.5279474)


    def test_structure_as_dict(self):

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        ).input_data(input_data=timeserie)
        hgf_dict = structure_as_dict(node_structure=hgf.node_trajectories)
        assert list(hgf_dict.keys()) == ['node_0', 'node_1', 'node_2']

    def test_structure_validation(self):
        
        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        structure_validation(hgf.node_structure)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
