# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
from jax.lax import scan
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.math import gaussian_surprise
from pyhgf.networks import beliefs_propagation
from pyhgf.typing import Indexes
from pyhgf.updates.posterior.continuous import (
    continuous_node_update,
    continuous_node_update_ehgf,
)
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction_error.inputs.continuous import (
    continuous_input_prediction_error,
)
from pyhgf.updates.prediction_error.nodes.continuous import (
    continuous_node_prediction_error,
)


class Testcontinuous(TestCase):
    def test_continuous_node_update(self):
        # create a node structure with no value parent and no volatility parent
        input_node_parameters = {
            "input_noise": 1e4,
            "expected_precision": jnp.nan,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "volatility_coupling_parents": None,
            "value_coupling_parents": None,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_1 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": None,
            "value_coupling_parents": None,
            "volatility_coupling_parents": None,
            "volatility_coupling_children": None,
            "autoregressive_coefficient": 0.0,
            "autoregressive_intercept": 0.0,
            "mean": 1.0,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_2 = {
            "expected_precision": 1.0,
            "precision": 1.0,
            "expected_mean": 1.0,
            "value_coupling_children": None,
            "value_coupling_parents": None,
            "volatility_coupling_parents": None,
            "volatility_coupling_children": None,
            "autoregressive_coefficient": 0.0,
            "autoregressive_intercept": 0.0,
            "mean": 1.0,
            "tonic_volatility": 1.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
        )
        edges = (
            Indexes(None, None, None, None),
            Indexes(None, None, None, None),
            Indexes(None, None, None, None),
        )
        data = jnp.array([0.2])
        time_steps = jnp.ones(1)

        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        sequence1 = 0, continuous_input_prediction_error
        update_sequence = (sequence1,)
        new_attributes, _ = beliefs_propagation(
            attributes=attributes,
            edges=edges,
            update_sequence=update_sequence,
            input_data=(data, time_steps),
        )

        assert attributes[1] == new_attributes[1]
        assert attributes[2] == new_attributes[2]

    def test_gaussian_surprise(self):
        surprise = gaussian_surprise(
            x=jnp.array([1.0, 1.0]),
            expected_mean=jnp.array([0.0, 0.0]),
            expected_precision=jnp.array([1.0, 1.0]),
        )
        assert jnp.all(jnp.isclose(surprise, 1.4189385))

    def test_continuous_input_update(self):
        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "input_noise": 1e4,
            "expected_precision": jnp.nan,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "volatility_coupling_parents": (1.0,),
            "value_coupling_parents": (1.0,),
            "temp": {
                "effective precision": 1.0,  # should be fixed to 1 for input nodes
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_1 = {
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
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_2 = {
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
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
        )

        edges = (
            Indexes((1,), None, None, None),
            Indexes(None, (2,), (0,), None),
            Indexes(None, None, None, (1,)),
        )

        # create update sequence
        sequence1 = 1, continuous_node_prediction
        sequence2 = 2, continuous_node_prediction
        sequence3 = 0, continuous_input_prediction_error
        sequence4 = 1, continuous_node_update
        sequence5 = 1, continuous_node_prediction_error
        sequence6 = 2, continuous_node_update
        update_sequence = (
            sequence1,
            sequence2,
            sequence3,
            sequence4,
            sequence5,
            sequence6,
        )
        data = jnp.array([0.2])
        time_steps = jnp.ones(1)

        # apply beliefs propagation updates
        new_attributes, _ = beliefs_propagation(
            edges=edges,
            attributes=attributes,
            update_sequence=update_sequence,
            input_data=(data, time_steps),
        )

        for idx, val in zip(["time_step", "value"], [1.0, 0.2]):
            assert jnp.isclose(new_attributes[0][idx], val)
        for idx, val in zip(
            ["precision", "expected_precision", "mean", "expected_mean"],
            [10000.119, 0.11920292, 0.20000952, 1.0],
        ):
            assert jnp.isclose(new_attributes[1][idx], val)
        for idx, val in zip(
            ["precision", "expected_precision", "mean", "expected_mean"],
            [0.34702963, 0.26894143, -0.17222318, 1.0],
        ):
            assert jnp.isclose(new_attributes[2][idx], val)

    def test_scan_loop(self):
        timeserie = load_data("continuous")

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "input_noise": 1e4,
            "expected_precision": jnp.nan,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "volatility_coupling_parents": None,
            "value_coupling_parents": (1.0,),
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_1 = {
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
            "tonic_volatility": -3.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        node_parameters_2 = {
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
            "tonic_volatility": -3.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }

        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
        )
        edges = (
            Indexes((1,), None, None, None),
            Indexes(None, (2,), (0,), None),
            Indexes(None, None, None, (1,)),
        )

        # create update sequence
        sequence1 = 2, continuous_node_prediction
        sequence2 = 1, continuous_node_prediction
        sequence3 = 0, continuous_input_prediction_error
        sequence4 = 1, continuous_node_update
        sequence5 = 1, continuous_node_prediction_error
        sequence6 = 2, continuous_node_update_ehgf
        update_sequence = (
            sequence1,
            sequence2,
            sequence3,
            sequence4,
            sequence5,
            sequence6,
        )

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=update_sequence,
            edges=edges,
        )

        # Create the data (value and time steps vectors)
        time_steps = jnp.ones((len(timeserie), 1))

        # Run the entire for loop
        last, _ = scan(scan_fn, attributes, (timeserie, time_steps))
        for idx, val in zip(["time_step", "value"], [1.0, 0.8241]):
            assert jnp.isclose(last[0][idx], val)
        for idx, val in zip(
            ["precision", "expected_precision", "mean", "expected_mean"],
            [24557.84, 14557.839, 0.8041823, 0.79050046],
        ):
            assert jnp.isclose(last[1][idx], val)
        for idx, val in zip(
            ["precision", "expected_precision", "mean", "expected_mean"],
            [1.3334407, 1.3493799, -7.1686087, -7.509615],
        ):
            assert jnp.isclose(last[2][idx], val)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
