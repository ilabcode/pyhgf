# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf import load_data
from pyhgf.model import HGF


class Testmodel(TestCase):
    def test_HGF(self):
        """Test the model class"""

        ##############
        # Continuous #
        ##############
        timeserie = load_data("continuous")

        # two-level
        # ---------
        two_level_continuous_hgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": timeserie[0], "2": 0.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -3.0, "2": -3.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        
        two_level_continuous_hgf.input_data(input_data=timeserie)

        surprise = two_level_continuous_hgf.surprise()  # Sum the surprise for this model
        assert jnp.isclose(surprise, -1941.3623)
        assert len(two_level_continuous_hgf.node_trajectories[0]["surprise"]) == 614

        # three-level
        # -----------
        three_level_continuous_hgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        )
        three_level_continuous_hgf.input_data(input_data=timeserie)
        surprise = three_level_continuous_hgf.surprise()
        assert jnp.isclose(surprise, -1918.408)


        ##########
        # Binary #
        ##########
        timeseries = load_data("binary")

        # two-level
        # ---------
        two_level_binary_hgf = HGF(
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
        two_level_binary_hgf = two_level_binary_hgf.input_data(timeseries)
        surprise = two_level_binary_hgf.surprise()
        assert jnp.isclose(surprise, 215.58821)

        # three-level
        # -----------
        three_level_binary_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mu={"1": .0, "2": .5, "3": 0.},
            initial_pi={"1": .0, "2": 1e4, "3": 1e1},
            omega={"1": None, "2": -6.0, "3": -2.0},
            rho={"1": None, "2": 0.0, "3": 0.0},
            kappas={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            pihat=jnp.inf,
        )
        three_level_binary_hgf.input_data(input_data=timeseries)
        surprise = three_level_binary_hgf.surprise()
        assert jnp.isclose(surprise, 215.59067)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
