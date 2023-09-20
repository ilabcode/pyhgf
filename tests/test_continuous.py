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
from pyhgf.updates.continuous import (
    continuous_input_prediction,
    continuous_input_prediction_error,
    continuous_node_prediction,
    continuous_node_prediction_error,
)


class Testcontinuous(TestCase):
    def test_continuous_node_update(self):
        # create a node structure with no value parent and no volatility parent
        input_node_parameters = {
            "pihat": 1e4,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "kappas_parents": None,
            "psis_parents": None,
        }
        node_parameters_1 = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "psis_children": None,
            "psis_parents": None,
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
            "psis_children": None,
            "psis_parents": None,
            "kappas_parents": None,
            "kappas_children": None,
            "mu": 1.0,
            "omega": 1.0,
            "rho": 0.0,
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
        data = jnp.array([0.2, 1.0])

        ###########################################
        # No value parent - no volatility parents #
        ###########################################
        sequence1 = 0, continuous_input_prediction
        sequence2 = 0, continuous_input_prediction_error
        update_sequence = (sequence1, sequence2)
        new_attributes, _ = beliefs_propagation(
            attributes=attributes,
            edges=edges,
            update_sequence=update_sequence,
            data=data,
        )

        assert attributes[1] == new_attributes[1]
        assert attributes[2] == new_attributes[2]

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
            "pihat": 1e4,
            "surprise": 0.0,
            "time_step": 0.0,
            "value": 0.0,
            "kappas_parents": (1.0,),
            "psis_parents": (1.0,),
        }
        node_parameters_1 = {
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
        node_parameters_2 = {
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
        sequence1 = 0, continuous_input_prediction
        sequence2 = 1, continuous_node_prediction
        sequence3 = 0, continuous_input_prediction_error
        sequence4 = 1, continuous_node_prediction_error
        update_sequence = (sequence1, sequence2, sequence3, sequence4)
        data = jnp.array([0.2, 1.0])

        # apply beliefs propagation updates
        new_attributes, _ = beliefs_propagation(
            edges=edges,
            attributes=attributes,
            update_sequence=update_sequence,
            data=data,
        )

        for idx, val in zip(["time_step", "value"], [1.0, 0.2]):
            assert jnp.isclose(new_attributes[0][idx], val)
        for idx, val in zip(
            ["pi", "pihat", "mu", "muhat"],
            [10000.119, 0.11920292, 0.20000952, 1.0],
        ):
            assert jnp.isclose(new_attributes[1][idx], val)
        for idx, val in zip(
            ["pi", "pihat", "mu", "muhat"],
            [0.29854316, 0.26894143, -0.36260414, 1.0],
        ):
            assert jnp.isclose(new_attributes[2][idx], val)

    def test_scan_loop(self):
        timeserie = load_data("continuous")

        # Create the data (value and time steps vectors)
        data = jnp.array([timeserie, jnp.ones(len(timeserie), dtype=int)]).T

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "pihat": 1e4,
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
            "psis_parents": None,
            "kappas_parents": (1.0,),
            "kappas_children": None,
            "mu": 1.0,
            "omega": -3.0,
            "rho": 0.0,
        }
        node_parameters_2 = {
            "pihat": 1.0,
            "pi": 1.0,
            "muhat": 1.0,
            "psis_children": None,
            "psis_parents": None,
            "kappas_parents": None,
            "kappas_children": (1.0,),
            "mu": 1.0,
            "omega": -3.0,
            "rho": 0.0,
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
        sequence1 = 0, continuous_input_prediction
        sequence2 = 1, continuous_node_prediction
        sequence3 = 0, continuous_input_prediction_error
        sequence4 = 1, continuous_node_prediction_error
        update_sequence = (sequence1, sequence2, sequence3, sequence4)

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=update_sequence,
            edges=edges,
        )

        # Run the entire for loop
        last, _ = scan(scan_fn, attributes, data)
        for idx, val in zip(["time_step", "value"], [1.0, 0.8241]):
            assert jnp.isclose(last[0][idx], val)
        for idx, val in zip(
            ["pi", "pihat", "mu", "muhat"],
            [22792.508, 12792.507, 0.80494785, 0.7899765],
        ):
            assert jnp.isclose(last[1][idx], val)
        for idx, val in zip(
            ["pi", "pihat", "mu", "muhat"],
            [1.4523009, 1.4297459, -6.9464974, -7.3045917],
        ):
            assert jnp.isclose(last[2][idx], val)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
