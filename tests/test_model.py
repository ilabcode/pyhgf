# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from ghgf import load_data
from ghgf.model import HGF


class Testmodel(TestCase):
    def test_HGF(self):
        """Test the model class"""

        ##############
        # Continuous #
        ##############

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        two_levels_continuous_hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        two_levels_continuous_hgf.input_data(input_data=timeserie)

        surprise = two_levels_continuous_hgf.surprise()  # Sum the surprise for this model
        assert jnp.isclose(surprise, -1938.0101)


        ##########
        # Binary #
        ##########

        timeserie = load_data("binary")

        two_levels_binary_hgf = HGF(
            n_levels=2,
            model_type="binary",
            initial_mu={"1": .0, "2": .5},
            initial_pi={"1": .0, "2": 1e4},
            omega={"1": None, "2": -6.0},
            rho={"1": None, "2": 0.0},
            kappas={"1": None},
            eta0=0.0,
            eta1=1.0,
            pihat = jnp.inf,
        )

        # Provide new observations
        two_levels_binary_hgf = two_levels_binary_hgf.input_data(timeserie)
        surprise = two_levels_binary_hgf.surprise()
        assert jnp.isclose(surprise, 215.11276)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
