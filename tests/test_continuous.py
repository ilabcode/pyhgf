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
    apply_sequence
)
from pyhgf.structure import structure_validation


class Testcontinuous(TestCase):

    def test_update_sequences(self):



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
            value_parents_idx=None,
            volatility_parents_idx=None,
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
            value_parents_idx=(2,),
            volatility_parents_idx=None,
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
            "volatility_parents": (2,),
            }
        node_structure = {1: node, 2: volatility_parent_node}

        new_node_structure = continuous_node_update(
            node_structure=node_structure,
            node_idx=1,
            value_parents_idx=None,
            volatility_parents_idx=(2,),
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
            "value_parents": (2,),
            "volatility_parents": (3,),
            }
        node_structure = {
            1: node, 
            2: value_parent_node,
            3: volatility_parent_node,
            }

        sequence1 = 1, (2,), (3,), continuous_node_update
        update_sequence = (sequence1, sequence1)

        apply_sequence(
            node_structure=node_structure, update_sequence=update_sequence, time_step=1.0
            )

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
            "value_parents": (1,),
            "volatility_parents": (2,),
            }
        node_structure = {
            0: input_node, 
            1: value_parent_node,
            2: volatility_parent_node,
            }

        node_structure = continuous_input_update(
            node_structure=node_structure,
            value=0.2,
            time_step=1.0,
            node_idx=0,
            value_parents_idx=(1,),
            volatility_parents_idx=(2,)
        )

        assert jnp.isclose(
            node_structure[0]["parameters"]["surprise"],
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
            "value_parents": (1,),
            "volatility_parents": (2,),
            }
        node_structure = {
            0: input_node, 
            1: value_parent_node,
            2: volatility_parent_node,
            }

        data = (5.0, 1.0)  # value, new_time

        final, last = loop_continuous_inputs(
            node_structure=node_structure, 
            data=data,
            )

        assert last[0]["parameters"]["time"] == 1.0
        assert last[0]["parameters"]["value"] == 5.0
        assert jnp.isclose(last[0]["parameters"]["surprise"], 2.5279474)


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
            "value_parents": [1],
            "volatility_parents": [2],
            }
        node_structure = {
            0: input_node, 
            1: value_parent_node,
            2: volatility_parent_node,
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
