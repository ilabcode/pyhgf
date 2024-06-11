# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.typing import AdjacencyLists, Inputs
from pyhgf.updates.posterior.continuous import (
    continuous_node_update,
    continuous_node_update_ehgf,
)
from pyhgf.updates.prediction_error.inputs.continuous import (
    continuous_input_prediction_error,
)
from pyhgf.utils import beliefs_propagation, list_branches


class TestNetworks(TestCase):
    def test_beliefs_propagation(self):
        """Test the loop_inputs function"""

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "input_precision": 1e4,
            "expected_precision": jnp.nan,
            "surprise": 0.0,
            "time_step": 0.0,
            "values": 0.0,
            "observed": 1,
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
            "mean": 1.0,
            "observed": 1,
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
            "mean": 1.0,
            "observed": 1,
            "tonic_volatility": -3.0,
            "tonic_drift": 0.0,
            "temp": {
                "effective_precision": 1.0,
                "value_prediction_error": 0.0,
                "volatility_prediction_error": 0.0,
            },
        }
        edges = (
            AdjacencyLists(0, (1,), None, None, None),
            AdjacencyLists(2, None, (2,), (0,), None),
            AdjacencyLists(2, None, None, None, (1,)),
        )
        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
        )

        # create update sequence
        sequence1 = 0, continuous_input_prediction_error
        sequence2 = 1, continuous_node_update
        sequence3 = 2, continuous_node_update_ehgf
        update_sequence = (sequence1, sequence2, sequence3)

        # one batch of new observations with time step
        data = jnp.array([0.2])
        time_steps = jnp.ones(1)
        observed = jnp.ones(1)
        inputs = Inputs(0, 1)

        # apply sequence
        new_attributes, _ = beliefs_propagation(
            attributes=attributes,
            input_data=(data, time_steps, observed),
            update_sequence=update_sequence,
            structure=(inputs, edges),
        )

        assert new_attributes[1]["mean"] == 0.20008
        assert new_attributes[2]["precision"] == 1.5

    def test_find_branch(self):
        """Test the find_branch function"""
        edges = (
            AdjacencyLists(0, (1,), None, None, None),
            AdjacencyLists(2, None, (2,), (0,), None),
            AdjacencyLists(2, None, None, None, (1,)),
            AdjacencyLists(2, (4,), None, None, None),
            AdjacencyLists(2, None, None, (3,), None),
        )
        branch_list = list_branches([0], edges, branch_list=[])
        assert branch_list == [0, 1, 2]

    def test_trim_sequence(self):
        """Test the trim_sequence function"""
        # TODO: need to rewrite the trim sequence method
        # edges = (
        #     Indexes((1,), None, None, None),
        #     Indexes(None, (2,), (0,), None),
        #     Indexes(None, None, None, (1,)),
        #     Indexes((4,), None, None, None),
        #     Indexes(None, None, (3,), None),
        # )
        # update_sequence = (
        #     (0, continuous_input_prediction_error),
        #     (1, continuous_node_prediction_error),
        #     (2, continuous_node_prediction_error),
        #     (3, continuous_node_prediction_error),
        #     (4, continuous_node_prediction_error),
        # )
        # new_sequence = trim_sequence(
        #     exclude_node_idxs=[0],
        #     update_sequence=update_sequence,
        #     edges=edges,
        # )
        # assert len(new_sequence) == 2
        # assert new_sequence[0][0] == 3
        # assert new_sequence[1][0] == 4


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
