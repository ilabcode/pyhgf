# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase
from jax.tree_util import Partial

import jax.numpy as jnp
from jax.lax import scan

from pyhgf import load_data
from pyhgf.binary import (
    binary_input_update,
    binary_node_update,
    binary_surprise,
    gaussian_density,
    sgm
)
from pyhgf.continuous import continuous_node_update
from pyhgf.structure import apply_sequence, loop_inputs
from pyhgf.typing import Indexes



class Testbinary(TestCase):

    def test_gaussian_density(self):
        surprise = gaussian_density(
            x=jnp.array([1.0, 1.0]),
            mu=jnp.array([0.0, 0.0]),
            pi=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 0.24197073))

    def test_sgm(self):
        assert jnp.all(jnp.isclose(sgm(jnp.array([.3, .3])), 0.5744425))

    def test_binary_surprise(self):
        surprise = binary_surprise(
            x=jnp.array([1.0]),
            muhat=jnp.array([0.2]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.609438))
        
    def test_update_binary_input_parents(self):

        ##########################
        # three level binary HGF #
        ##########################
        input_node_parameters = {
            "pihat": jnp.inf,
            "eta0": 0.0,
            "eta1": 1.0,
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
            "rho": 0.0,
        }

        node_structure = (
            Indexes((1,), None),
            Indexes((2,), None),
            Indexes(None, (3,)),
            Indexes(None, None),
        )
        parameters_structure = (
            input_node_parameters,
            node_parameters,
            node_parameters,
            node_parameters,
        )
        
        # create update sequence
        sequence1 = 0, binary_input_update
        sequence2 = 1, binary_node_update
        sequence3 = 2, continuous_node_update
        update_sequence = (sequence1, sequence2, sequence3)

        # apply sequence
        new_parameters_structure = apply_sequence(
            node_structure=node_structure,
            parameters_structure=parameters_structure,
            update_sequence=update_sequence, 
            time_step=1.0,
            values=jnp.array([1.0]),
            input_nodes_idx=jnp.array([0])
            )
        assert new_parameters_structure[0]["surprise"] == 0.31326166

        # use scan
        timeserie = load_data("binary")

        # Create the data (value and time steps vectors)
        data = jnp.array([timeserie, jnp.ones(len(timeserie), dtype=int)]).T

        # create the function that will be scaned
        scan_fn = Partial(
            loop_inputs, 
            update_sequence=update_sequence, 
            node_structure=node_structure,
            input_nodes_idx=jnp.array([0])
            )

        # Run the entire for loop
        last, final = scan(scan_fn, parameters_structure, data)

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
