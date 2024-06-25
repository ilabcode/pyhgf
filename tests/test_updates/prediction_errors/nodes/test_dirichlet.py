# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.model import Network
from pyhgf.updates.prediction_error.nodes.dirichlet import (
    dirichlet_node_prediction_error,
    get_candidate,
)


class TestDirichletNode(TestCase):
    def test_get_candidate(self):
        mean, precision = get_candidate(
            value=5.0,
            sensory_precision=1.0,
            expected_mean=jnp.array([0.0, -5.0]),
            expected_sigma=jnp.array([1.0, 3.0]),
        )

        assert jnp.isclose(mean, 5.026636)
        assert jnp.isclose(precision, 1.2752448)

    def test_dirichlet_node_prediction_error(self):

        network = (
            Network()
            .add_nodes(kind="generic-input")
            .add_nodes(kind="DP-state", value_children=0)
            .add_nodes(
                kind="ef-normal",
                n_nodes=2,
                value_children=1,
                xis=jnp.array([0.0, 1 / 8]),
                nus=15.0,
            )
        )

        attributes, (_, edges), _ = network.get_network()
        dirichlet_node_prediction_error(
            edges=edges,
            attributes=attributes,
            node_idx=1,
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
