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
from pyhgf.typing import Indexes

class Testcontinuous(TestCase):

    def test_update_sequences(self):

        # create a node structure with one value parent and one volatility parent
        node_parameters = {
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
        input_parameters = {
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
            Indexes(None, None),
            Indexes(None, None),
            Indexes(None, None),
        )

        parameters_structure = (
            input_parameters,
            node_parameters,
            node_parameters,
        )

        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure = apply_sequence(
            parameters_structure=parameters_structure, 
            time_step=1.0,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            value=None
            )

        assert parameters_structure == new_parameters_structure

        #######################
        # x_1 as value parent #
        #######################
        node_structure = (
            Indexes((1,), None),
            Indexes(None, None),
            Indexes(None, None),
        )

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure = apply_sequence(
            parameters_structure=parameters_structure, 
            time_step=1.0,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            value=None
            )

        assert jnp.isclose(
            new_parameters_structure[1]["pi"],
            1.2689414
        )

        ############################
        # x_1 as volatility parent #
        ############################
        node_structure = (
            Indexes(None, (1,)),
            Indexes(None, None),
            Indexes(None, None),
        )

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            parameters_structure=parameters_structure, 
            time_step=1.0,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            value=None
            )
        assert jnp.isclose(
            new_node_structure[1]["pi"],
            0.7689414
        )

        #####################################
        # Both value and volatility parents #
        #####################################
        node_structure = (
            Indexes((1,), (2,)),
            Indexes(None, None),
            Indexes(None, None),
        )

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_node_structure = apply_sequence(
            parameters_structure=parameters_structure, 
            time_step=1.0,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            value=None
            )
        assert jnp.isclose(
            new_node_structure[1]["pi"],
            1.2689414
        )
        assert jnp.isclose(
            new_node_structure[2]["pi"],
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

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": None,
            "psis": None,
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
            value=.2
            )

        assert new_parameters_structure[1]["pi"] == 0.48708236

    def test_scan_loop(self):

        timeserie = load_data("continuous")

        # Create the data (value and time steps vectors)
        data = jnp.array([timeserie, jnp.ones(len(timeserie), dtype=int)]).T

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "omega": 1.0,
            "kappas": None,
            "psis": None,
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

        # create the function that will be scaned
        scan_fn = Partial(
            loop_inputs, 
            update_sequence=update_sequence, 
            node_structure=node_structure
            )

        # Run the entire for loop
        last, final = scan(scan_fn, parameters_structure, data)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
