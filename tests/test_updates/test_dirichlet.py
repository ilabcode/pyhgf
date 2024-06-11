# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.updates.prediction_error.nodes.dirichlet import get_candidate


class TestDirichlet(TestCase):
    def test_get_candidate(self):
        mean, precision = get_candidate(
            value=5.0,
            sensory_precision=1.0,
            expected_mean=jnp.array([0.0, -5.0]),
            expected_sigma=jnp.array([1.0, 3.0]),
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
