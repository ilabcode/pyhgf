# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from unittest import TestCase

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ghgf import load_data
from ghgf.model import HGF
from ghgf.python import StandardHGF, StandardBinaryHGF


class TestCompareBackends(TestCase):

    def test_compare_backends_continuous(self):
        """Compare the JAX and pure Python implementation of the HGF."""

        np.random.seed(123)
        timeseries = load_data("continuous")

        ################
        # 2-levels HGF #
        ################
        jaxhgf = HGF(
            n_levels=2,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        jaxhgf.input_data(input_data=timeseries)
        jax_surprise = jaxhgf.results["surprise"].sum()

        stdhgf = StandardHGF(
            n_levels=2,
            process_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappas={"1": 1.0},
        )
        stdhgf.input(timeseries)
        std_surprise = stdhgf.surprise()

        assert jnp.isclose(jax_surprise, -1922.2264)
        assert jnp.isclose(std_surprise, -1924.5794736619316)
        assert len(stdhgf.input_nodes[0].surprises) == 615
        assert len(jaxhgf.results["surprise"]) == 613

        ############
        # Surprise #
        ############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.input_nodes[0].surprises[1:], label="Standard HGF")
        plt.plot(jaxhgf.results["surprise"], label="JAX HGF")
        plt.legend()

        ##############
        # First node #
        ##############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x1.mus, label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][0]["mu"], label="JAX HGF")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x2.mus, label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][0]["mu"], label="JAX HGF")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x2.pis, label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][0]["pi"], label="JAX HGF")
        plt.legend()

        ################
        # 3-levels HGF #
        ################
        jaxhgf = HGF(
            n_levels=3,
            model_type="continuous",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        )
        jaxhgf.input_data(input_data=timeseries)
        jax_surprise = jaxhgf.results["surprise"].sum()

        stdhgf = StandardHGF(
            n_levels=3,
            process_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
            initial_pi={"1": 1e4, "2": 1e1, "3": 1e1},
            omega={"1": -13.0, "2": -2.0, "3": -2.0},
            rho={"1": 0.0, "2": 0.0, "3": 0.0},
            kappas={"1": 1.0, "2": 1.0},
        )
        stdhgf.input(timeseries)
        std_surprise = stdhgf.surprise()

        assert jnp.isclose(jax_surprise, -1915.0765)
        assert jnp.isclose(std_surprise, -1914.917339132975)

        ############
        # Surprise #
        ############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.input_nodes[0].surprises[2:], label="Standard HGF")
        plt.plot(jaxhgf.results["surprise"], label="JAX HGF")
        plt.title("Surprise")
        plt.legend()

        ##############
        # First node #
        ##############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x1.mus[2:], label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_1$")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x2.mus[2:], label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_2$")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x2.pis, label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][0]["pi"], label="JAX HGF")
        plt.legend()

        ##############
        # Third node #
        ##############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x3.mus[2:], label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][2][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_3$")
        plt.legend()

        ##############
        # Third node #
        ##############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x3.pis, label="Standard HGF")
        plt.plot(jaxhgf.node_trajectories[1][0][2][0][2][0][0]["pi"], label="JAX HGF")
        plt.legend()

    def test_compare_backends_binary(self):
        """Compare the JAX and pure Python implementation of the HGF."""

        np.random.seed(123)
        timeseries = load_data("binary")

        stdhgf = StandardBinaryHGF(
            initial_mu2=.5,
            initial_pi2=1e4,
            initial_mu3=0,
            initial_pi3=1e1,
            omega2=-6,
            omega3=-2,
            kappa2=1,
        )
        stdhgf.input(timeseries)
        assert stdhgf.surprise() == 215.5906767857954

        jaxhgf = HGF(
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
        jaxhgf.input_data(input_data=timeseries)
        jax_surprise = jaxhgf.results["surprise"].sum()

        assert jnp.isclose(jax_surprise, 215.11488)
        assert jnp.isclose(stdhgf.surprise(), 215.5906767857954)

        ############
        # Surprise #
        ############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.input_nodes[0].surprises[2:], label="Binary HGF - Python")
        plt.plot(jaxhgf.results["surprise"], label="JAX HGF")
        plt.title("Surprise")
        plt.legend()

        ##############
        # First node #
        ##############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x1.mus[2:], label="Binary HGF - Python")
        plt.plot(jaxhgf.node_trajectories[1][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_1$")
        plt.legend()
        
        assert sum(stdhgf.x1.mus[2:]) - sum(jaxhgf.node_trajectories[1][0][0]["mu"]) == 0

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x2.mus[2:], label="Binary HGF - Python")
        plt.plot(jaxhgf.node_trajectories[1][0][1][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_2$")
        plt.legend()
        
        assert sum(stdhgf.x2.mus[2:]) - sum(jaxhgf.node_trajectories[1][0][1][0][0]["mu"]) < .6

        ##############
        # Third node #
        ##############
        plt.figure(figsize=(12, 3))
        plt.plot(stdhgf.x3.mus[2:], label="Binary HGF - Python")
        plt.plot(jaxhgf.node_trajectories[1][0][1][0][2][0][0]["mu"], label="JAX HGF")
        plt.title(r"$\mu_3$")
        plt.legend()

        assert sum(stdhgf.x3.mus[2:]) - sum(jaxhgf.node_trajectories[1][0][1][0][2][0][0]["mu"]) < .04
