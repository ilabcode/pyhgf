# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
import numpy as np

from ghgf import load_data
from ghgf.model import HGF


class Testmodel(TestCase):
    def test_HGF(self):
        """Test the model class"""

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")
        data = np.array([timeserie, np.arange(1, len(timeserie) + 1, dtype=float)]).T

        jaxhgf = HGF(
            n_levels=2,
            model_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        jaxhgf.input_data(input_data=data)

        surprise = jaxhgf.surprise()  # Sum the surprise for this model
        assert jnp.isclose(surprise, -1922.2264)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
