# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.binary import (
    binary_input_update,
    binary_node_update,
    binary_surprise,
    gaussian_density,
    sgm,
)
from pyhgf.continuous import continuous_node_update
from pyhgf.structure import beliefs_propagation
from pyhgf.typing import Indexes
import numpy as np

class Testbinary(TestCase):
    def test_gaussian_density(self):
        surprise = gaussian_density(
            x=jnp.array([1.0, 1.0]),
            mu=jnp.array([0.0, 0.0]),
            pi=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 0.24197073))

    def test_sgm(self):
        assert jnp.all(jnp.isclose(sgm(jnp.array([0.3, 0.3])), 0.5744425))

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
        data = jnp.array([1.0, 1.0])

        # apply sequence
        new_parameters_structure, _ = beliefs_propagation(
            node_structure=node_structure,
            parameters_structure=parameters_structure,
            update_sequence=update_sequence,
            data=data,
        )
        for idx, val in zip(["surprise", "value"], [0.31326166, 1.0]):
            assert jnp.isclose(new_parameters_structure[0][idx], val)
        for idx, val in zip(["mu", "muhat", "pihat"], [1.0, 0.7310586, 5.0861616]):
            assert jnp.isclose(new_parameters_structure[1][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat", "nu"],
            [1.8515793, 1.0, 0.31581485, 0.11920292, 7.389056],
        ):
            assert jnp.isclose(new_parameters_structure[2][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat", "nu"],
            [0.5611493, 1.0, 0.5380009, 0.26894143, 2.7182817],
        ):
            assert jnp.isclose(new_parameters_structure[3][idx], val)

        # use scan
        timeserie = load_data("binary")

        # Create the data (value and time steps vectors) - only use the 5 first trials
        # as the priors are ill defined here
        data = jnp.array([timeserie, jnp.ones(len(timeserie), dtype=int)]).T[:5]

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=update_sequence,
            node_structure=node_structure,
        )

        # Run the entire for loop
        last, _ = scan(scan_fn, parameters_structure, data)
        for idx, val in zip(["surprise", "value"], [3.1497865, 0.0]):
            assert jnp.isclose(last[0][idx], val)
        for idx, val in zip(["mu", "muhat", "pihat"], [0.0, 0.9571387, 24.37586]):
            assert jnp.isclose(last[1][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat", "nu"],
            [-2.358439, 3.1059794, 0.17515838, 0.13413419, 2.1602433],
        ):
            assert jnp.isclose(last[2][idx], val)
        for idx, val in zip(
            ["muhat", "pihat", "nu"], [-0.22977911, 0.14781797, 2.7182817]
        ):
            assert jnp.isclose(last[3][idx], val)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
