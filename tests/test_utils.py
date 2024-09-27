# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
from pytest import raises

from pyhgf import load_data
from pyhgf.model import Network
from pyhgf.typing import AdjacencyLists
from pyhgf.utils import list_branches


def test_imports():
    """Test the data import function"""
    _ = load_data("continuous")
    _, _ = load_data("binary")

    with raises(Exception):
        load_data("error")


def test_add_edges():
    """Test the add_edges function."""
    network = Network().add_nodes().add_nodes(n_nodes=3)
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
        .add_nodes(kind="binary-state")
        .add_nodes(value_children=0)
        .create_belief_propagation_fn()
    )

    predictions, updates = network1.update_sequence
    assert len(predictions) == 2
    assert len(updates) == 2

    # a standard continuous HGF
    network2 = (
        Network()
        .add_nodes()
        .add_nodes(value_children=0)
        .add_nodes(volatility_children=1)
        .create_belief_propagation_fn(update_type="standard")
    )
    predictions, updates = network2.update_sequence
    assert len(predictions) == 3
    assert len(updates) == 4

    # a generic input with a normal-EF node
    network3 = (
        Network()
        .add_nodes(kind="generic-state")
        .add_nodes(kind="ef-normal", value_children=0)
        .create_belief_propagation_fn()
    )
    predictions, updates = network3.update_sequence
    assert len(predictions) == 0
    assert len(updates) == 2

    # a Dirichlet node
    network4 = (
        Network()
        .add_nodes(kind="generic-state")
        .add_nodes(kind="DP-state", value_children=0, alpha=0.1, batch_size=2)
        .add_nodes(
            kind="ef-normal",
            n_nodes=2,
            value_children=1,
            xis=jnp.array([0.0, 1 / 8]),
            nus=15.0,
        )
        .create_belief_propagation_fn()
    )
    predictions, updates = network4.update_sequence
    assert len(predictions) == 1
    assert len(updates) == 4
