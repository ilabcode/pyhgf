# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.math import binary_surprise, gaussian_density, sigmoid
from pyhgf.typing import AdjacencyLists, Inputs
from pyhgf.updates.posterior.binary import binary_node_update_infinite
from pyhgf.updates.posterior.continuous import continuous_node_update
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction_error.inputs.binary import (
    binary_input_prediction_error_infinite_precision,
)
from pyhgf.updates.prediction_error.nodes.binary import (
    binary_state_node_prediction_error,
)
from pyhgf.updates.prediction_error.nodes.continuous import (
    continuous_node_prediction_error,
)
from pyhgf.utils import beliefs_propagation


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
            "values": 0.0,
            "observed": 1,
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
            "autoconnection_strength": 1.0,
            "mean": 1.0,
            "observed": 1,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "binary_expected_precision": jnp.nan,
            "temp": {
                "value_prediction_error": 0.0,
            },
        }
        node_parameters_2 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": (1.0,),
            "value_coupling_parents": None,
            "volatility_coupling_parents": (1.0,),
            "volatility_coupling_children": None,
            "autoconnection_strength": 1.0,
            "mean": 1.0,
            "observed": 1,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_3 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": None,
            "value_coupling_parents": None,
            "volatility_coupling_parents": None,
            "volatility_coupling_children": (1.0,),
            "autoconnection_strength": 1.0,
            "mean": 1.0,
            "observed": 1,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }

        edges = (
            AdjacencyLists(0, (1,), None, None, None),
            AdjacencyLists(1, (2,), None, (0,), None),
            AdjacencyLists(2, None, (3,), (1,), None),
            AdjacencyLists(2, None, None, None, (2,)),
        )
        attributes = {
            0: input_node_parameters,
            1: node_parameters_1,
            2: node_parameters_2,
            3: node_parameters_3,
        }

        # create update sequence
        sequence1 = 3, continuous_node_prediction
        sequence2 = 2, continuous_node_prediction
        sequence3 = 1, binary_state_node_prediction
        sequence4 = 0, binary_input_prediction_error_infinite_precision
        sequence5 = 1, binary_node_update_infinite
        sequence6 = 1, binary_state_node_prediction_error
        sequence7 = 2, continuous_node_update
        sequence8 = 2, continuous_node_prediction_error
        sequence9 = 3, continuous_node_update
        update_sequence = (
            sequence1,
            sequence2,
            sequence3,
            sequence4,
            sequence5,
            sequence6,
            sequence7,
            sequence8,
            sequence9,
        )
        data = jnp.ones(1)
        time_steps = jnp.ones(1)
        observed = jnp.ones(1)
        inputs = Inputs(0, 1)

        # apply sequence
        new_attributes, _ = beliefs_propagation(
            structure=(inputs, edges),
            attributes=attributes,
            update_sequence=update_sequence,
            input_data=(data, time_steps, observed),
        )
        for idx, val in zip(
            ["mean", "expected_mean", "binary_expected_precision"],
            [1.0, 0.7310586, 5.0861616],
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
        data = jnp.array([u[:5]]).T
        time_steps = jnp.ones((len(u[:5]), 1))
        observed = jnp.ones((len(u[:5]), 1))
        inputs = Inputs(0, 1)

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=update_sequence,
            structure=(inputs, edges),
        )

        # Run the entire for loop
        last, _ = scan(scan_fn, attributes, (data, time_steps, observed))
        for idx, val in zip(
            ["mean", "expected_mean", "binary_expected_precision"],
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
