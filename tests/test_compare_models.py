# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from unittest import TestCase

import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy import loadtxt

from ghgf.hgf import StandardHGF
from ghgf.model import HGF


class Testsdt(TestCase):
    def test_models_comparison(self):

        timeserie = loadtxt("/home/nicolas/git/ghgf/tests/data/usdchf.dat")
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

        jaxhgf = HGF(
            n_levels=2,
            model_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappa={"1": 1.0},
        )
        jaxhgf.input_data(input_data=data)

        node, results = jaxhgf.final
        jax_surprise = results["surprise"].sum()

        stdhgf = StandardHGF(
            n_levels=2,
            model_type="GRW",
            initial_mu={"1": 1.04, "2": 1.0},
            initial_pi={"1": 1e4, "2": 1e1},
            omega={"1": -13.0, "2": -2.0},
            rho={"1": 0.0, "2": 0.0},
            kappa={"1": 1.0},
        )
        stdhgf.input(timeserie)
        std_surprise = stdhgf.surprise()

        assert jax_surprise == -1925.712
        assert std_surprise == -1924.5794736619316
        len(stdhgf.input_nodes[0].surprises)
        len(results["surprise"])

        ############
        # Surprise #
        ############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.input_nodes[0].surprises[1:], label="Standard HGF")
        plt.plot(results["surprise"], label="JAX HGF")
        plt.legend()

        ##############
        # First node #
        ##############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x1.mus, label="Standard HGF")
        plt.plot(node[1][0]["mu"], label="JAX HGF")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x2.mus, label="Standard HGF")
        plt.plot(node[1][2][0][0]["mu"], label="JAX HGF")
        plt.legend()

        ###############
        # Second node #
        ###############
        plt.figure(figsize=(12, 6))
        plt.plot(stdhgf.x2.pis, label="Standard HGF")
        plt.plot(node[1][2][0][0]["pi"], label="JAX HGF")
        plt.legend()
