# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest

import jax.numpy as jnp
import pytest
from jax.lax import scan
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.math import gaussian_surprise
from pyhgf.typing import AdjacencyLists, Inputs
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
from pyhgf.utils import beliefs_propagation


@pytest.fixture(scope="module")
def nodes_attributes():
    input_node_parameters = {
        "input_precision": 1e4,
        "expected_precision": 1e4,
        "surprise": 0.0,
        "time_step": 0.0,
        "values": 0.0,
        "observed": 1,
        "volatility_coupling_parents": (1.0,),
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
        "value_coupling_children": (1.0,),
        "value_coupling_parents": None,
        "volatility_coupling_parents": (1.0,),
        "volatility_coupling_children": None,
        "autoconnection_strength": 1.0,
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
        "autoconnection_strength": 1.0,
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
    attributes = {
        0: input_node_parameters,
        1: node_parameters_1,
        2: node_parameters_2,
    }
    return attributes


def test_continuous_node_update(nodes_attributes):
    # create a node structure with no value parent and no volatility parent
    attributes = nodes_attributes
    edges = (
        AdjacencyLists(0, None, None, None, None, (None,)),
        AdjacencyLists(2, None, None, None, None, (None,)),
        AdjacencyLists(2, None, None, None, None, (None,)),
    )
    data = jnp.array([0.2])
    time_steps = jnp.ones(1)
    observed = jnp.ones(1)
    inputs = Inputs(0, 0)

    ###########################################
    # No value parent - no volatility parents #
    ###########################################
    sequence1 = 0, continuous_input_prediction_error
    update_sequence = (sequence1,)
    new_attributes, _ = beliefs_propagation(
        attributes=attributes,
        structure=(inputs, edges),
        update_sequence=update_sequence,
        input_data=(data, time_steps, observed),
    )

    assert attributes[1] == new_attributes[1]
    assert attributes[2] == new_attributes[2]


def test_gaussian_surprise():
    surprise = gaussian_surprise(
        x=jnp.array([1.0, 1.0]),
        expected_mean=jnp.array([0.0, 0.0]),
        expected_precision=jnp.array([1.0, 1.0]),
    )
    assert jnp.all(jnp.isclose(surprise, 1.4189385))


def test_continuous_input_update(nodes_attributes):
    ###############################################
    # one value parent with one volatility parent #
    ###############################################
    attributes = nodes_attributes
    def identity(x):
        return(x)
    
    edges = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (None,)),
    )

    # define a non linear network behaving like the linear one
    edges_nonlinear = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (identity,)),
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
    observed = jnp.ones(1)
    inputs = Inputs(0, 0)

    # apply beliefs propagation updates
    new_attributes, _ = beliefs_propagation(
        structure=(inputs, edges),
        attributes=attributes,
        update_sequence=update_sequence,
        input_data=(data, time_steps, observed),
    )

    #repeat for non-linear network
    new_attributes_nonlinear, _ = beliefs_propagation(
        structure=(inputs, edges_nonlinear),
        attributes=attributes,
        update_sequence=update_sequence,
        input_data=(data, time_steps, observed),
    )

    for idx, val in zip(["time_step", "values"], [1.0, 0.2]):
        assert jnp.isclose(new_attributes[0][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [10000.881, 0.880797, 0.20007047, 1.0],
    ):
        assert jnp.isclose(new_attributes[1][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [0.9794834, 0.95257413, 0.97345114, 1.0],
    ):
        assert jnp.isclose(new_attributes[2][idx], val)

    for idx, val in zip(["time_step", "values"], [1.0, 0.2]):
        assert jnp.isclose(new_attributes_nonlinear[0][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [10000.881, 0.880797, 0.20007047, 1.0],
    ):
        assert jnp.isclose(new_attributes_nonlinear[1][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [0.9794834, 0.95257413, 0.97345114, 1.0],
    ):
        assert jnp.isclose(new_attributes_nonlinear[2][idx], val)

def test_continuous_input_update_nonlinear(nodes_attributes):
    ###############################################
    # one value parent with one volatility parent #
    ###############################################
    attributes = nodes_attributes
    def identity(x):
        return(x)

    # define a non linear network behaving like the linear one
    edges_nonlinear = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (identity,)),
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
    observed = jnp.ones(1)
    inputs = Inputs(0, 0)

    # apply beliefs propagation updates
    new_attributes_nonlinear, _ = beliefs_propagation(
        structure=(inputs, edges_nonlinear),
        attributes=attributes,
        update_sequence=update_sequence,
        input_data=(data, time_steps, observed),
    )

    for idx, val in zip(["time_step", "values"], [1.0, 0.2]):
        assert jnp.isclose(new_attributes_nonlinear[0][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [10000.881, 0.880797, 0.20007047, 1.0],
    ):
        assert jnp.isclose(new_attributes_nonlinear[1][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [0.9794834, 0.95257413, 0.97345114, 1.0],
    ):
        assert jnp.isclose(new_attributes_nonlinear[2][idx], val)


def test_scan_loop(nodes_attributes):
    timeserie = load_data("continuous")

    ###############################################
    # one value parent with one volatility parent #
    ###############################################
    attributes = nodes_attributes
    edges = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (None,)),
    )
    
    #testing it with a non-linear coupling
    def identity(x):
        return(x)
    
    edges_nonlinear = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (identity,)),
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

    inputs = Inputs(0, 1)

    # create the function that will be scaned
    scan_fn = Partial(
        beliefs_propagation,
        update_sequence=update_sequence,
        structure=(inputs, edges),
    )

    scan_fn_nonlinear = Partial(
        beliefs_propagation,
        update_sequence=update_sequence,
        structure=(inputs, edges),
    )

    # Create the data (value and time steps vectors)
    time_steps = jnp.ones((len(timeserie), 1))
    observed = jnp.ones((len(timeserie), 1))

    # Run the entire for loop
    last, _ = scan(scan_fn, attributes, (timeserie, time_steps, observed))
    for idx, val in zip(["time_step", "values"], [1.0, 0.8241]):
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

    # non linear coupling
        # Run the entire for loop
    last_nonlinear, _ = scan(scan_fn_nonlinear, attributes, 
                   (timeserie, time_steps, observed))
    for idx, val in zip(["time_step", "values"], [1.0, 0.8241]):
        assert jnp.isclose(last_nonlinear[0][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [24557.84, 14557.839, 0.8041823, 0.79050046],
    ):
        assert jnp.isclose(last_nonlinear[1][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.3334407, 1.3493799, -7.1686087, -7.509615],
    ):
        assert jnp.isclose(last_nonlinear[2][idx], val)

def test_scan_loop_nonlinear(nodes_attributes):
    timeserie = load_data("continuous")

    ###############################################
    # one value parent with one volatility parent #
    ###############################################
    attributes = nodes_attributes

    #defining an identity function
    def identity(x):
        return(x)
    
    edges_nonlinear = (
        AdjacencyLists(0, (1,), None, None, None, (None,)),
        AdjacencyLists(2, None, (2,), (0,), None, (None,)),
        AdjacencyLists(2, None, None, None, (1,), (identity,)),
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

    inputs = Inputs(0, 1)

    # create the function that will be scaned
    scan_fn_nonlinear = Partial(
        beliefs_propagation,
        update_sequence=update_sequence,
        structure=(inputs, edges_nonlinear),
    )

    # Create the data (value and time steps vectors)
    time_steps = jnp.ones((len(timeserie), 1))
    observed = jnp.ones((len(timeserie), 1))

    # Run the entire for loop
    last_nonlinear, _ = scan(scan_fn_nonlinear, attributes, 
                   (timeserie, time_steps, observed))
    for idx, val in zip(["time_step", "values"], [1.0, 0.8241]):
        assert jnp.isclose(last_nonlinear[0][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [24557.84, 14557.839, 0.8041823, 0.79050046],
    ):
        assert jnp.isclose(last_nonlinear[1][idx], val)
    for idx, val in zip(
        ["precision", "expected_precision", "mean", "expected_mean"],
        [1.3334407, 1.3493799, -7.1686087, -7.509615],
    ):
        assert jnp.isclose(last_nonlinear[2][idx], val)


def test_coupling_fn_multiple_children():
    """ Tests if the coupling function is passed correctly
    into the network"""

    # creating a simple coupling function
    def identity(x):
        return x
    
    # I create a network with a node with 2 value children
    test_HGF = (
        Network()
        .add_nodes(kind="continuous-input")
        .add_nodes(kind="continuous-input")
        .add_nodes(value_children=0, n_nodes=1)
        .add_nodes(value_children=[1, 2], n_nodes=1, 
                   coupling_fn=(None,identity)
        ) 
    )

    # check if the number of coupling fn matches the number of children
    coupling_fn_length = []
    children_number = []
    for node_idx in range(2,len(test_HGF.edges)):
        coupling_fn_length.append(len(test_HGF.edges[node_idx].coupling_fn))
        children_number.append(len(test_HGF.edges[node_idx].value_children))

    assert children_number == coupling_fn_length



if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
