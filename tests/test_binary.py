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
            mean=jnp.array([0.0, 0.0]),
            precision=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 0.24197073))

    def test_sgm(self):
        assert jnp.all(jnp.isclose(sigmoid(jnp.array([0.3, 0.3])), 0.5744425))

    def test_binary_surprise(self):
        surprise = binary_surprise(
            x=jnp.array([1.0]),
            expected_mean=jnp.array([0.2]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.609438))

    def test_update_binary_input_parents(self):
        ##########################
        # three level binary HGF #
        ##########################
        input_node_parameters = {
            "expected_precision": jnp.inf,
            "eta0": 0.0,
            "eta1": 1.0,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "volatility_coupling_parents": None,
            "value_coupling_parents": (1.0,),
        }
        node_parameters_1 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": (1.0,),
            "value_coupling_parents": (1.0,),
            "volatility_coupling_parents": None,
            "volatility_coupling_children": None,
            "autoregressive_coefficient": 0.0,
            "autoregressive_intercept": 0.0,
            "mean": 1.0,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {"predicted_volatility": 0.0},
        }
        node_parameters_2 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": (1.0,),
            "value_coupling_parents": None,
            "volatility_coupling_parents": (1.0,),
            "volatility_coupling_children": None,
            "autoregressive_coefficient": 0.0,
            "autoregressive_intercept": 0.0,
            "mean": 1.0,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {"predicted_volatility": 0.0},
        }
        node_parameters_3 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": None,
            "value_coupling_parents": None,
            "volatility_coupling_parents": None,
            "volatility_coupling_children": (1.0,),
            "autoregressive_coefficient": 0.0,
            "autoregressive_intercept": 0.0,
            "mean": 1.0,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {"predicted_volatility": 0.0},
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
        sequence1 = 3, continuous_node_prediction
        sequence2 = 2, continuous_node_prediction
        sequence3 = 1, binary_node_prediction
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
        for idx, val in zip(
            ["mean", "expected_mean", "expected_precision"], [1.0, 0.7310586, 5.0861616]
        ):
            assert jnp.isclose(new_attributes[1][idx], val)
        for idx, val in zip(
            ["mean", "expected_mean", "precision", "expected_precision"],
            [1.8515793, 1.0, 0.31581485, 0.11920292],
        ):
            assert jnp.isclose(new_attributes[2][idx], val)
        for idx, val in zip(
            ["mean", "expected_mean", "precision", "expected_precision"],
            [0.5050575, 1.0, 0.47702926, 0.26894143],
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
        for idx, val in zip(["surprise", "value"], [3.1274157, 0.0]):
            assert jnp.isclose(last[0][idx], val)
        for idx, val in zip(
            ["mean", "expected_mean", "expected_precision"],
            [0.0, 0.95616907, 23.860779],
        ):
            assert jnp.isclose(last[1][idx], val)
        for idx, val in zip(
            ["mean", "expected_mean", "precision", "expected_precision"],
            [-2.1582031, 3.0825963, 0.18244718, 0.1405374],
        ):
            assert jnp.isclose(last[2][idx], val)
        for idx, val in zip(
            ["expected_mean", "expected_precision"], [-0.30260748, 0.14332297]
        ):
            assert jnp.isclose(last[3][idx], val)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
