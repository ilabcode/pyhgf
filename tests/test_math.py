# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf.math import (
    MultivariateNormal,
    Normal,
    binary_surprise_finite_precision,
    gaussian_predictive_distribution,
    sigmoid_inverse_temperature,
)


class TestMath(TestCase):
    def test_multivariate_normal(self):

        ss = MultivariateNormal.sufficient_statistics(jnp.array([1.0, 2.0]))
        assert jnp.isclose(
            ss, jnp.array([1.0, 2.0, 1.0, 2.0, 4.0], dtype="float32")
        ).all()

        bm = MultivariateNormal.base_measure(2)
        assert bm == 0.15915494309189535

    def test_normal(self):

        ss = Normal.sufficient_statistics(jnp.array(1.0))
        assert jnp.isclose(ss, jnp.array([1.0, 1.0], dtype="float32")).all()

        bm = Normal.base_measure()
        assert bm == 0.3989423

        ess = Normal.expected_sufficient_statistics(mu=0.0, sigma=1.0)
        assert jnp.isclose(ess, jnp.array([0.0, 1.0], dtype="float32")).all()

        par = Normal.parameters(xis=[5.0, 29.0])
        assert jnp.isclose(jnp.array(par), jnp.array([5.0, 4.0], dtype="float32")).all()

    def test_gaussian_predictive_distribution(self):

        pdf = gaussian_predictive_distribution(x=1.5, xi=[0.0, 1 / 8], nu=5.0)
        assert jnp.isclose(pdf, jnp.array(0.00845728, dtype="float32"))

    def test_binary_surprise_finite_precision(self):

        surprise = binary_surprise_finite_precision(
            value=1.0,
            expected_mean=0.0,
            expected_precision=1.0,
            eta0=0.0,
            eta1=1.0,
        )
        assert surprise == 1.4189385

    def test_sigmoid_inverse_temperature(self):
        s = sigmoid_inverse_temperature(x=0.4, temperature=6.0)
        assert jnp.isclose(s, jnp.array(0.08070617906683485, dtype="float32"))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
