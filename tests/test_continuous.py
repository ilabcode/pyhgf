# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase
from collections import namedtuple
import jax.numpy as jnp
from jax.lax import scan

from pyhgf import load_data
from pyhgf.continuous import (
    gaussian_surprise,
    loop_continuous_inputs,
    continuous_input_update,
    continuous_node_update,
)
from pyhgf.structure import structure_validation


class Testcontinuous(TestCase):

    def test_continuous_node_update(self):

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
        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        node = {
            "parameters": parameters,
            "value_parents": None,
            "volatility_parents": None,
            }
        node_structure = {1: node}

        new_node_structure = continuous_node_update(
            node_structure=node_structure,
            node_idx=1,
            time_step=1.0
        )

        assert node_structure == new_node_structure

        #######################
        # x_2 as value parent #
        #######################
        input_parameters = {
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
        value_parent_node = {
            "parameters": parameters,
            "value_parents": None,
            "volatility_parents": None
        }
        node = {
            "parameters": input_parameters,
            "value_parents": (2,),
            "volatility_parents": None,
            }
        node_structure = {1: node, 2: value_parent_node}

        new_node_structure = continuous_node_update(
            node_structure=node_structure,
            node_idx=1,
            time_step=1.0
        )
        assert jnp.isclose(
            new_node_structure[2]["parameters"]["pi"],
            1.2689414
        )

        ############################
        # x_2 as volatility parent #
        ############################
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
        volatility_parent_node = {
            "parameters": parameters,
            "value_parents": None,
            "volatility_parents": None
        }
        node = {
            "parameters": node_parameters,
            "value_parents": None,
            "volatility_parents": [2],
            }
        node_structure = {1: node, 2: volatility_parent_node}

        new_node_structure = continuous_node_update(
            node_structure=node_structure,
            node_idx=1,
            time_step=1.0
        )
        assert jnp.isclose(
            new_node_structure[2]["parameters"]["pi"],
            0.7689414
        )

        #####################################
        # Both value and volatility parents #
        #####################################
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
        node = {
            "parameters": node_parameters,
            "value_parents": [2],
            "volatility_parents": [3],
            }
        node_structure = {
            1: node, 
            2: value_parent_node,
            3: volatility_parent_node,
            }

        new_node_structure = continuous_node_update(
            node_structure=node_structure,
            node_idx=1,
            time_step=1.0
        )

        # ########################################################################
        # # Both value and volatility parents with values and volatility parents #
        # ########################################################################

        # # Second level
        # vo_pa_2 = {
        #     "mu": jnp.array(1.0),
        #     "muhat": jnp.array(1.0),
        #     "pi": jnp.array(1.0),
        #     "pihat": jnp.array(1.0),
        #     "kappas": None,
        #     "nu": jnp.array(1.0),
        #     "psis": None,
        #     "omega": jnp.array(-2.0),
        #     "rho": jnp.array(0.0),
        # }
        # vo_pa_3 = {
        #     "mu": jnp.array(1.0),
        #     "muhat": jnp.array(1.0),
        #     "pi": jnp.array(1.0),
        #     "pihat": jnp.array(1.0),
        #     "kappas": None,
        #     "nu": jnp.array(1.0),
        #     "psis": None,
        #     "omega": jnp.array(-2.0),
        #     "rho": jnp.array(0.0),
        # }
        # volatility_parent_2 = vo_pa_2, None, None
        # volatility_parent_3 = vo_pa_3, None, None

        # va_pa_2 = {
        #     "mu": jnp.array(1.0),
        #     "muhat": jnp.array(1.0),
        #     "pi": jnp.array(1.0),
        #     "pihat": jnp.array(1.0),
        #     "kappas": (jnp.array(1.0),),
        #     "nu": jnp.array(1.0),
        #     "psis": None,
        #     "omega": jnp.array(-2.0),
        #     "rho": jnp.array(0.0),
        # }
        # value_parent_2 = va_pa_2, None, (volatility_parent_3,)

        # # First level
        # vo_pa_1 = {
        #     "mu": jnp.array(1.0),
        #     "muhat": jnp.array(1.0),
        #     "pi": jnp.array(1.0),
        #     "pihat": jnp.array(1.0),
        #     "kappas": None,
        #     "nu": jnp.array(1.0),
        #     "psis": (jnp.array(1.0),),
        #     "omega": jnp.array(-2.0),
        #     "rho": jnp.array(0.0),
        # }
        # va_pa = {
        #     "mu": jnp.array(1.0),
        #     "muhat": jnp.array(1.0),
        #     "pi": jnp.array(1.0),
        #     "pihat": jnp.array(1.0),
        #     "kappas": (jnp.array(1.0),),
        #     "nu": jnp.array(1.0),
        #     "psis": None,
        #     "omega": jnp.array(-2.0),
        #     "rho": jnp.array(0.0),
        # }
        # volatility_parent_1 = vo_pa_1, (value_parent_2,), None
        # value_parent_1 = va_pa, None, (volatility_parent_2,)

        # node_parameters, value_parents, volatility_parents = volatility_parent_1
        # output_node = continuous_node_update(
        #     node_parameters=node_parameters,
        #     value_parents=value_parents,
        #     volatility_parents=volatility_parents,
        #     old_time=1,
        #     new_time=2,
        # )

        # # Verify node structure
        # structure_validation(output_node)
        # assert len(output_node) == 3
        # assert isinstance(output_node[0], dict)
        # assert isinstance(output_node[1], tuple)
        # assert output_node[2] is None

        # node_parameters, value_parents, volatility_parents = value_parent_1

        # output_node = continuous_node_update(
        #     node_parameters=node_parameters,
        #     value_parents=value_parents,
        #     volatility_parents=volatility_parents,
        #     old_time=1,
        #     new_time=2,
        # )

        # # Verify node structure
        # structure_validation(output_node)
        # assert len(output_node) == 3
        # assert isinstance(output_node[0], dict)
        # assert output_node[1] is None
        # assert isinstance(output_node[2], tuple)

    def test_gaussian_surprise(self):
        surprise = gaussian_surprise(
            x=jnp.array([1.0, 1.0]),
            muhat=jnp.array([0.0, 0.0]),
            pihat=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.4189385))

    def test_update_continuous_input_parents(self):

        ############################################
        # one value parent - one volatility parent #
        ############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": 1.0,
            "surprise": 0.0,
            "time": 0.0,
            "value": 0.0,
        }
        parameters = {
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
            "value_parents": [2],
            "volatility_parents": [3],
            }
        node_structure = {
            "input_node_0": input_node, 
            2: value_parent_node,
            3: volatility_parent_node,
            }

        node_structure = continuous_input_update(
            node_structure=node_structure,
            value=0.2,
            time_step=1.0,
        )

        assert jnp.isclose(
            node_structure["input_node_0"]["parameters"]["surprise"],
            2.1381817
        )

    def test_loop_continuous_inputs(self):
        """Test the function that should be scanned"""

        ############################################
        # one value parent - one volatility parent #
        ############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": 1.0,
            "surprise": 0.0,
            "time": 0.0,
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
            "value_parents": [2],
            "volatility_parents": [3],
            }
        node_structure = {
            "input_node_0": input_node, 
            2: value_parent_node,
            3: volatility_parent_node,
            }

        data = (5.0, 1.0)  # value, new_time

        final, last = loop_continuous_inputs(node_structure=node_structure, data=data)

        assert last["input_node_0"]["parameters"]["time"] == 1.0
        assert last["input_node_0"]["parameters"]["value"] == 5.0
        assert jnp.isclose(last["input_node_0"]["parameters"]["surprise"], 2.5279474)


    def test_scan_loop(self):

        ##################
        # Continuous HGF #
        ##################

        timeserie = load_data("continuous")

        input_node_parameters = {
            "omega": jnp.array(1.0),
            "kappas": jnp.array(1.0),
            "surprise": jnp.array(0.0),
            "time": jnp.array(0.0),
            "value": jnp.array(0.0),
        }
        parameters = {
            "pihat": jnp.array(1.0),
            "pi": jnp.array(1.0),
            "muhat": jnp.array(1.0),
            "kappas": None,
            "mu": jnp.array(1.0),
            "nu": jnp.array(1.0),
            "psis": (jnp.array(1.0),),
            "omega": jnp.array(1.0),
            "rho": jnp.array(1.0),
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
            "value_parents": [2],
            "volatility_parents": [3],
            }
        node_structure = {
            "input_node_0": input_node, 
            2: value_parent_node,
            3: volatility_parent_node,
            }
            
        for val, t in zip(timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)):
            data = (val, t)
            final, last = loop_continuous_inputs(node_structure=node_structure, data=data)

        # # Create the data (value and time vectors)
        # data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        # # Run the entire for loop
        # last, final = scan(loop_continuous_inputs, node_structure, data)




if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
