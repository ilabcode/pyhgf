# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.networks import beliefs_propagation, list_branches, trim_sequence
from pyhgf.typing import Indexes
from pyhgf.updates.continuous import (
    continuous_input_prediction_error,
    continuous_node_prediction_error,
)


class TestStructure(TestCase):
    def test_beliefs_propagation(self):
        """Test the loop_inputs function"""

        ###############################################
        # one value parent with one volatility parent #
        ###############################################
        input_node_parameters = {
            "expected_precision": 1e4,
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
            "value_coupling_parents": None,
            "volatility_coupling_parents": (1.0,),
            "volatility_coupling_children": None,
            "mean": 1.0,
            "tonic_volatility": -3.0,
            "tonic_drift": 0.0,
            "temp": {"predicted_volatility": 0.0},
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
            "tonic_volatility": -3.0,
            "tonic_drift": 0.0,
            "temp": {"predicted_volatility": 0.0},
        }
        edges = (
            Indexes((1,), None, None, None),
            Indexes(None, (2,), (0,), None),
            Indexes(None, None, None, (1,)),
        )
        attributes = (
            input_node_parameters,
            node_parameters_1,
            node_parameters_2,
        )

        # create update sequence
        sequence1 = 0, continuous_input_prediction_error
        sequence2 = 1, continuous_node_prediction_error
        update_sequence = (sequence1, sequence2)

        # one batch of new observations with time step
        data = jnp.array([0.2, 1.0])

        # apply sequence
        new_attributes, _ = beliefs_propagation(
            attributes=attributes,
            data=data,
            update_sequence=update_sequence,
            edges=edges,
        )

        assert new_attributes[1]["mean"] == 0.20007998
        assert new_attributes[2]["precision"] == 1.0

    def test_find_branch(self):
        """Test the find_branch function"""
        edges = (
            Indexes((1,), None, None, None),
            Indexes(None, (2,), (0,), None),
            Indexes(None, None, None, (1,)),
            Indexes((4,), None, None, None),
            Indexes(None, None, (3,), None),
        )
        branch_list = list_branches([0], edges, branch_list=[])
        assert branch_list == [0, 1, 2]

    def test_trim_sequence(self):
        """Test the trim_sequence function"""
        edges = (
            Indexes((1,), None, None, None),
            Indexes(None, (2,), (0,), None),
            Indexes(None, None, None, (1,)),
            Indexes((4,), None, None, None),
            Indexes(None, None, (3,), None),
        )
        update_sequence = (
            (0, continuous_input_prediction_error),
            (1, continuous_node_prediction_error),
            (2, continuous_node_prediction_error),
            (3, continuous_node_prediction_error),
            (4, continuous_node_prediction_error),
        )
        new_sequence = trim_sequence(
            exclude_node_idxs=[0],
            update_sequence=update_sequence,
            edges=edges,
        )
        assert len(new_sequence) == 2
        assert new_sequence[0][0] == 3
        assert new_sequence[1][0] == 4


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
