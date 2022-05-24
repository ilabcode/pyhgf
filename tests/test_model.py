# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
from unittest import TestCase

import arviz as az
import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
from jax import random
from numpy import loadtxt
from numpyro.infer import MCMC, NUTS

from ghgf.model import HGF, HGFDistribution


class Testmodel(TestCase):
    def test_HGF(self):
        """Test the model class"""

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")
        data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T

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

    def test_HGFDistribution(self):
        """Test the model distribution"""

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = jnp.array(
            [
                jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T
                for _ in range(3)
            ]
        )

        # Simulate recording with different lenght
        input_data = input_data.at[1, 610:, :].set(jnp.nan)
        input_data = input_data.at[2, 600:, :].set(jnp.nan)

        rho_1 = jnp.array([0.0, 0.0, 0.0])
        rho_2 = jnp.array([0.0, 0.0, 0.0])
        pi_1 = jnp.array([1e4, 1e4, 1e4])
        pi_2 = jnp.array([1e1, 1e1, 1e1])
        mu_1 = jnp.array(
            [input_data[0][0][0], input_data[0][0][0], input_data[0][0][0]]
        )
        mu_2 = jnp.array([0.0, 0.0, 0.0])
        kappa = jnp.array([1.0, 1.0])

        hgf = HGFDistribution(
            input_data=input_data,
            omega_1=jnp.array([-3.0, -3.0, -3.0]),
            omega_2=jnp.array([-3.0, -3.0, -3.0]),
            rho_1=rho_1,
            rho_2=rho_2,
            pi_1=pi_1,
            pi_2=pi_2,
            mu_1=mu_1,
            mu_2=mu_2,
            kappa_1=kappa,
        )

        assert jnp.isclose(hgf.log_prob(None), 5765.3843)

        #################
        # Test sampling #
        #################

        input_data = jnp.array(
            [
                jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T
                for _ in range(3)
            ]
        )

        def model(input_data):

            # Hyper-parameters
            μ_ω_1 = npy.sample("μ_ω_1", dist.Normal(0.0, 2.0))
            σ_ω_1 = npy.sample("σ_ω_1", dist.HalfNormal(0.5))

            with npy.plate("plate_i", 3):
                ω_1 = npy.sample("ω_1", dist.Normal(μ_ω_1, σ_ω_1))

            omega_2 = jnp.array([-3.0, -3.0, -3.0])
            rho_1 = jnp.array([0.0, 0.0, 0.0])
            rho_2 = jnp.array([0.0, 0.0, 0.0])
            pi_1 = jnp.array([1e4, 1e4, 1e4])
            pi_2 = jnp.array([1e1, 1e1, 1e1])
            mu_1 = jnp.array(
                [input_data[0][0][0], input_data[0][0][0], input_data[0][0][0]]
            )
            mu_2 = jnp.array([0.0, 0.0, 0.0])
            kappa_1 = jnp.array([1.0, 1.0, 1.0])

            npy.sample(
                "hgf_log_prob",
                HGFDistribution(
                    input_data=input_data,
                    omega_1=ω_1,
                    omega_2=omega_2,
                    rho_1=rho_1,
                    rho_2=rho_2,
                    pi_1=pi_1,
                    pi_2=pi_2,
                    mu_1=mu_1,
                    mu_2=mu_2,
                    kappa_1=kappa_1,
                ),
            )

        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model)
        num_samples = 1000
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
        mcmc.run(rng_key_, input_data=input_data)

        samples = az.from_numpyro(mcmc)

        assert round(az.summary(samples)["mean"]["μ_ω_1"]) == 4


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
