# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
from pytest import raises

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists, Inputs
from pyhgf.updates.posterior.continuous import (
    continuous_node_update,
    continuous_node_update_ehgf,
)
from pyhgf.updates.prediction_error.inputs.continuous import (
    continuous_input_prediction_error,
)
from pyhgf.utils import beliefs_propagation, list_branches


def test_imports():
    """Test the data import function"""
    _ = load_data("continuous")
    _, _ = load_data("binary")

    with raises(Exception):
        load_data("error")


def test_beliefs_propagation():
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
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (None,)),
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

def test_add_edges():
    """Test the add_edges function."""
    network = Network().add_nodes(kind="continuous-input").add_nodes(n_nodes=3)
    with raises(Exception):
        network.add_edges(kind="error")

    network.add_edges(
        kind="volatility", parent_idxs=2, children_idxs=0, coupling_strengths=1
    )
    network.add_edges(parent_idxs=1, children_idxs=0, coupling_strengths=1.0)

def test_find_branch():
    """Test the find_branch function."""
    edges = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (None,)),
        AdjacencyLists(2, (4,), None, None, None, (None,)),
        AdjacencyLists(2, None, None, (3,), None, (None,)),
    )
    branch_list = list_branches([0], edges, branch_list=[])
    assert branch_list == [0, 1, 2]

def test_set_update_sequence():
    """Test the set_update_sequence function."""

    # a standard binary HGF
    network1 = (
        Network()
        .add_nodes(kind="binary-input")
        .add_nodes(kind="binary-state", value_children=0)
        .add_nodes(value_children=1)
        .set_update_sequence()
    )
    assert len(network1.update_sequence) == 6

    # a standard continuous HGF
    network2 = (
        Network()
        .add_nodes(kind="continuous-input")
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .set_update_sequence(update_type="standard")
    )
    assert len(network2.update_sequence) == 6

    # a generic input with a normal-EF node
    network3 = (
        Network()
        .add_nodes(kind="generic-input")
        .add_nodes(kind="ef-normal")
        .set_update_sequence()
    )
    assert len(network3.update_sequence) == 2

    # a Dirichlet node
    network4 = (
        Network()
        .add_nodes(kind="generic-input")
        .add_nodes(kind="DP-state", value_children=0, alpha=0.1, batch_size=2)
        .add_nodes(
            kind="ef-normal",
            n_nodes=2,
            value_children=1,
            xis=jnp.array([0.0, 1 / 8]),
            nus=15.0,
        )
        .set_update_sequence()
    )
    assert len(network4.update_sequence) == 5
