# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase
import jax.numpy as jnp
from jax.lax import scan

from pyhgf import load_data
from pyhgf.continuous import (
    gaussian_surprise,
    continuous_input_update,
    continuous_node_update,
)
from pyhgf.structure import loop_inputs, apply_sequence
from jax.tree_util import Partial


class Testcontinuous(TestCase):

    def test_update_sequences(self):

        # create a node structure with one value parent and one volatility parent
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
            "value_parents": (1,),
            "volatility_parents": (2,),
            }
        node_structure = {
            0: node, 
            1: value_parent_node,
            2: volatility_parent_node,
            }


        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        sequence1 = 0, None, None, continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            node_structure=node_structure, 
            update_sequence=update_sequence, 
            time_step=1.0
            )

        assert node_structure == new_node_structure

        #######################
        # x_1 as value parent #
        #######################
        sequence1 = 0, (1,), None, continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            node_structure=node_structure, 
            update_sequence=update_sequence, 
            time_step=1.0
            )

        assert jnp.isclose(
            new_node_structure[1]["parameters"]["pi"],
            1.2689414
        )

        ############################
        # x_1 as volatility parent #
        ############################
        sequence1 = 0, None, (1,), continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            node_structure=node_structure, 
            update_sequence=update_sequence, 
            time_step=1.0
            )
        assert jnp.isclose(
            new_node_structure[1]["parameters"]["pi"],
            0.7689414
        )

        #####################################
        # Both value and volatility parents #
        #####################################
        sequence1 = 0, (1,), (2,), continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            node_structure=node_structure, 
            update_sequence=update_sequence, 
            time_step=1.0
            )
        assert jnp.isclose(
            new_node_structure[1]["parameters"]["pi"],
            1.2689414
        )
        assert jnp.isclose(
            new_node_structure[2]["parameters"]["pi"],
            0.7689414
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
        
        # apply update steps
        sequence1 = 0, (1,), (2,), continuous_input_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            node_structure=node_structure, 
            update_sequence=update_sequence, 
            time_step=1.0,
            value=.2
            )

        assert jnp.isclose(
            new_node_structure[0]["parameters"]["surprise"],
            2.1381817
        )

    def test_scan_loop(self):

        ##################
        # Continuous HGF #
        ##################

        timeserie = load_data("continuous")

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
            "omega": -6.0,
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

        # Create the data (value and time steps vectors)
        data = jnp.array([timeserie, jnp.ones(len(timeserie), dtype=int)]).T
        
        sequence1 = 0, (1,), (2,), continuous_input_update
        sequence2 = 1, None, None, continuous_node_update
        sequence3 = 2, None, None, continuous_node_update
        update_sequence = (sequence1, sequence2, sequence3)

        # create the function that will be scaned
        scan_fn = Partial(loop_inputs, update_sequence)

        # Run the entire for loop
        last, final = scan(scan_fn, node_structure, data)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
