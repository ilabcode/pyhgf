import os
import unittest
from unittest import TestCase

import numpy as np

from ghgf.hgf import StandardHGF
from ghgf.plots import plot_trajectories

path = os.path.dirname(os.path.abspath(__file__))


class Testsdt(TestCase):
    def test_plot_trajectories(self):

        # Set up standard 2-level HGF for continuous inputs
        stdhgf = StandardHGF(
            n_levels=2,
            model_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappa={"1": 1.0},
        )

        # Read USD-CHF data
        usdchf = np.loadtxt(f"{path}/data/usdchf.dat")

        # Feed input
        stdhgf.input(usdchf)

        # Plot
        plot_trajectories(model=stdhgf, ci=True, figsize=800)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
