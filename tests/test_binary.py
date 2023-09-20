# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.math import binary_surprise, gaussian_density, sigmoid
from pyhgf.networks import beliefs_propagation
from pyhgf.typing import Indexes
from pyhgf.updates.binary import (
    binary_input_prediction,
    binary_input_prediction_error,
    binary_node_prediction,
    binary_node_prediction_error,
)
from pyhgf.updates.continuous import (
    continuous_node_prediction,
    continuous_node_prediction_error,
)


class Testbinary(TestCase):
    def test_gaussian_density(self):
        surprise = gaussian_density(
            x=jnp.array([1.0, 1.0]),
            mu=jnp.array([0.0, 0.0]),
            pi=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 0.24197073))

    def test_sgm(self):
        assert jnp.all(jnp.isclose(sigmoid(jnp.array([0.3, 0.3])), 0.5744425))

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
            "kappas_parents": None,
            "psis_parents": (1.0,),
        }
        node_parameters_1 = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "psis_children": (1.0,),
            "psis_parents": (1.0,),
            "kappas_parents": None,
            "kappas_children": None,
            "mu": 1.0,
            "omega": 1.0,
            "rho": 0.0,
        }
        node_parameters_2 = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "psis_children": (1.0,),
            "psis_parents": None,
            "kappas_parents": (1.0,),
            "kappas_children": None,
            "mu": 1.0,
            "omega": 1.0,
            "rho": 0.0,
        }
        node_parameters_3 = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "psis_children": None,
            "psis_parents": None,
            "kappas_parents": None,
            "kappas_children": (1.0,),
            "mu": 1.0,
            "omega": 1.0,
            "rho": 0.0,
        }

        edges = (
            Indexes((1,), None, None, None),
            Indexes((2,), None, (0,), None),
            Indexes(None, (3,), (1,), None),
            Indexes(None, None, None, (2,)),
        )
        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
            node_parameters_3,
        )

        # create update sequence
        sequence1 = 0, binary_input_prediction
        sequence2 = 1, binary_node_prediction
        sequence3 = 2, continuous_node_prediction
        sequence4 = 0, binary_input_prediction_error
        sequence5 = 1, binary_node_prediction_error
        sequence6 = 2, continuous_node_prediction_error
        update_sequence = (
            sequence1,
            sequence2,
            sequence3,
            sequence4,
            sequence5,
            sequence6,
        )
        data = jnp.array([1.0, 1.0])

        # apply sequence
        new_attributes, _ = beliefs_propagation(
            edges=edges,
            attributes=attributes,
            update_sequence=update_sequence,
            data=data,
        )
        for idx, val in zip(["surprise", "value"], [0.31326166, 1.0]):
            assert jnp.isclose(new_attributes[0][idx], val)
        for idx, val in zip(["mu", "muhat", "pihat"], [1.0, 0.7310586, 5.0861616]):
            assert jnp.isclose(new_attributes[1][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat"],
            [1.8515793, 1.0, 0.31581485, 0.11920292],
        ):
            assert jnp.isclose(new_attributes[2][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat"],
            [0.5611493, 1.0, 0.5380009, 0.26894143],
        ):
            assert jnp.isclose(new_attributes[3][idx], val)

        # use scan
        u, _ = load_data("binary")

        # Create the data (value and time steps vectors) - only use the 5 first trials
        # as the priors are ill defined here
        data = jnp.array([u, jnp.ones(len(u), dtype=int)]).T[:5]

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=update_sequence,
            edges=edges,
        )

        # Run the entire for loop
        last, _ = scan(scan_fn, attributes, data)
        for idx, val in zip(["surprise", "value"], [3.1497865, 0.0]):
            assert jnp.isclose(last[0][idx], val)
        for idx, val in zip(["mu", "muhat", "pihat"], [0.0, 0.9571387, 24.37586]):
            assert jnp.isclose(last[1][idx], val)
        for idx, val in zip(
            ["mu", "muhat", "pi", "pihat"],
            [-2.358439, 3.1059794, 0.17515838, 0.13413419],
        ):
            assert jnp.isclose(last[2][idx], val)
        for idx, val in zip(["muhat", "pihat"], [-0.22977911, 0.14781797]):
            assert jnp.isclose(last[3][idx], val)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
