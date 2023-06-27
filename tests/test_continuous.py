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
from pyhgf.structure import beliefs_propagation
from jax.tree_util import Partial
from pyhgf.typing import Indexes

class Testcontinuous(TestCase):

    def test_continuous_node_update(self):

        # create a node structure with no value parent and no volatility parent
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
        data = jnp.array([jnp.nan, 1.0])

        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure, _ = beliefs_propagation(
            parameters_structure=parameters_structure, 
            data=data,
            node_structure=node_structure,
            update_sequence=update_sequence, 
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
        data = jnp.array([jnp.nan, 1.0])

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure, _ = beliefs_propagation(
            parameters_structure=parameters_structure, 
            node_structure=node_structure,
            data=data,
            update_sequence=update_sequence, 
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [1.2689414, 0.26894143, 2.0, 2.0, 2.7182817]):
            assert jnp.isclose(
                new_parameters_structure[1][idx],
                val
            )

        ############################
        # x_1 as volatility parent #
        ############################
        node_structure = (
            Indexes(None, (1,)),
            Indexes(None, None),
            Indexes(None, None),
        )
        data = jnp.array([jnp.nan, 1.0])

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure, _ = beliefs_propagation(
            parameters_structure=parameters_structure, 
            data=data,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.7689414, 0.26894143, 2.0, 2.0, 2.7182817]):
            assert jnp.isclose(
                new_parameters_structure[1][idx],
                val
            )

        #####################################
        # Both value and volatility parents #
        #####################################
        node_structure = (
            Indexes((1,), (2,)),
            Indexes(None, None),
            Indexes(None, None),
        )
        data = jnp.array([jnp.nan, 1.0])

        sequence1 = 0, continuous_node_update
        update_sequence = (sequence1,)
        new_parameters_structure, _ = beliefs_propagation(
            parameters_structure=parameters_structure, 
            data=data,
            node_structure=node_structure,
            update_sequence=update_sequence, 
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [1.2689414, 0.26894143, 2.0, 2.0, 2.7182817]):
            assert jnp.isclose(
                new_parameters_structure[1][idx],
                val
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.7689414, 0.26894143, 2.0, 2.0, 2.7182817]):
            assert jnp.isclose(
                new_parameters_structure[2][idx],
                val
            )

    def test_gaussian_surprise(self):
        surprise = gaussian_surprise(
            x=jnp.array([1.0, 1.0]),
            muhat=jnp.array([0.0, 0.0]),
            pihat=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.4189385))

    def test_continuous_input_update(self):

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
        data = jnp.array([.2, 1.0])

        # apply beliefs propagation updates
        new_parameters_structure, _ = beliefs_propagation(
            node_structure=node_structure,
            parameters_structure=parameters_structure,
            update_sequence=update_sequence, 
            data=data,
            )

        for idx, val in zip(["time_step", "value"], [1.0, 0.2]):
            assert jnp.isclose(
                new_parameters_structure[0][idx],
                val
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.48708236, 0.11920292, 0.6405112, 2.0, 7.389056]):
            assert jnp.isclose(
                new_parameters_structure[1][idx],
                val
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.50698835, 0.26894143, 1.5353041, 2.0, 2.7182817]):
            assert jnp.isclose(
                new_parameters_structure[2][idx],
                val
            )

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
            beliefs_propagation, 
            update_sequence=update_sequence, 
            node_structure=node_structure,
            )

        # Run the entire for loop
        last, _ = scan(scan_fn, parameters_structure, data)
        for idx, val in zip(["time_step", "value"], [1.0, 0.8241]):
            assert jnp.isclose(
                last[0][idx],
                val
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.44606116, 0.07818171, 1.0302161, 2.000083, 10.548521]):
            assert jnp.isclose(
                last[1][idx],
                val
            )
        for idx, val in zip(["pi", "pihat", "mu", "muhat", "nu"], [0.30642906, 0.16752565, 1.3451389, 2.3559856, 2.7182817]):
            assert jnp.isclose(
                last[2][idx],
                val
            )

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
