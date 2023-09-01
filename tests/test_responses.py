# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from pyhgf import load_data
from pyhgf.model import HGF
from pyhgf.response import binary_softmax


class Testmodel(TestCase):
    def test_binary_softmax(self):
        u, y = load_data("binary")

        # two-level
        # ---------
        two_level_binary_hgf = HGF(
            n_levels=2,
            model_type="binary",
            initial_mu={"1": 0.0, "2": 0.5},
            initial_pi={"1": 0.0, "2": 1e4},
            omega={"1": None, "2": -6.0},
            rho={"1": None, "2": 0.0},
            kappas={"1": None},
            eta0=0.0,
            eta1=1.0,
            binary_precision=jnp.inf,
        ).input_data(input_data=u)

        surprise = two_level_binary_hgf.surprise(
            response_function=binary_softmax, response_function_parameters=y
        )
        assert np.isclose(surprise, 200.24422)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
