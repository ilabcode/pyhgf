# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from ghgf import load_data
from ghgf.model import HGF


class Testsdt(TestCase):
    def test_plot_trajectories(self):

        # Set up standard 3-level HGF for continuous inputs
        hgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        )

        # Read USD-CHF data
        timeserie = load_data("continuous")
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        # Feed input
        hgf.input_data(input_data=data)

        # Plot
        hgf.plot_trajectories()

        # Read binary input
        timeserie = load_data("binary")
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T
        three_levels_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mu={"1": .0, "2": .5, "3": 0.},
            initial_pi={"1": .0, "2": 1e4, "3": 1e1},
            omega={"1": None, "2": -6.0, "3": -2.0},
            rho={"1": None, "2": 0.0, "3": 0.0},
            kappas={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            pihat = jnp.inf,
        )
        
        # Feed input
        three_levels_hgf = three_levels_hgf.input_data(data)
        
        # Plot
        three_levels_hgf.plot_trajectories()

if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
