# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp

from pyhgf import load_data
from pyhgf.model import HGF


class Testplots(TestCase):
    def test_plotting_functions(self):
        # Read USD-CHF data
        timeserie = load_data("continuous")

        ##############
        # Continuous #
        # ------------

        # Set up standard 2-level HGF for continuous inputs
        two_level_continuous = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        ).input_data(input_data=timeserie)

        # plot trajectories
        two_level_continuous.plot_trajectories()

        # plot correlations
        two_level_continuous.plot_correlations()

        # plot node structures
        two_level_continuous.plot_network()

        # plot nodes
        two_level_continuous.plot_nodes(
            node_idxs=2, show_current_state=True, show_observations=True
        )

        # Set up standard 3-level HGF for continuous inputs
        three_level_continuous = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        ).input_data(input_data=timeserie)

        # plot trajectories
        three_level_continuous.plot_trajectories()

        # plot correlations
        three_level_continuous.plot_correlations()

        # plot node structures
        three_level_continuous.plot_network()

        # plot nodes
        three_level_continuous.plot_nodes(
            node_idxs=2, show_current_state=True, show_observations=True
        )

        ##########
        # Binary #
        # --------

        # Read binary input
        timeserie = load_data("binary")

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
        ).input_data(timeserie)

        # plot trajectories
        two_level_binary_hgf.plot_trajectories()

        # plot correlations
        two_level_binary_hgf.plot_correlations()

        # plot node structures
        two_level_binary_hgf.plot_network()

        # plot node structures
        two_level_binary_hgf.plot_nodes(
            node_idxs=2, show_current_state=True, show_observations=True
        )

        three_level_binary_hgf = HGF(
            n_levels=3,
            model_type="binary",
            initial_mu={"1": 0.0, "2": 0.5, "3": 0.0},
            initial_pi={"1": 0.0, "2": 1e4, "3": 1e1},
            omega={"1": None, "2": -6.0, "3": -2.0},
            rho={"1": None, "2": 0.0, "3": 0.0},
            kappas={"1": None, "2": 1.0},
            eta0=0.0,
            eta1=1.0,
            binary_precision=jnp.inf,
        ).input_data(timeserie)

        # plot trajectories
        three_level_binary_hgf.plot_trajectories()

        # plot correlations
        three_level_binary_hgf.plot_correlations()

        # plot node structures
        three_level_binary_hgf.plot_network()

        # plot node structures
        three_level_binary_hgf.plot_nodes(
            node_idxs=2, show_current_state=True, show_observations=True
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
