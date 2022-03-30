# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
from jax import random
from numpy import loadtxt
from numpyro.infer import MCMC, NUTS

from ghgf.model import HGF, HGFDistribution


class Testsdt(TestCase):
    def test_HGF(self):
        """Test the model class"""

        # Create the data (value and time vectors)
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

        surprise = jaxhgf.surprise()  # Sum the surprise for this model
        assert surprise == -1922.2264

    def test_HGFDistribution(self):
        """Test the model distribution"""

        def model(input_data):

            omega_1 = npy.sample("omega_1", dist.Normal(-10.0, 4.0), sample_shape=(1,))
            omega_2 = npy.sample("omega_2", dist.Normal(-10.0, 4.0), sample_shape=(1,))
            rho_1 = jnp.array([0.0])
            rho_2 = jnp.array([0.0])
            pi_1 = jnp.array([1e4])
            pi_2 = jnp.array([1e1])
            mu_1 = jnp.array([input_data[0][0]])
            mu_2 = jnp.array([1.0])
            kappa = jnp.array([1.0])

            npy.sample(
                "hgf_log_prob",
                HGFDistribution(
                    input_data=[input_data],
                    omega_1=omega_1,
                    omega_2=omega_2,
                    rho_1=rho_1,
                    rho_2=rho_2,
                    pi_1=pi_1,
                    pi_2=pi_2,
                    mu_1=mu_1,
                    mu_2=mu_2,
                    kappa=kappa,
                ),
            )

        # Create the data (value and time vectors)
        timeserie = loadtxt("/home/nicolas/git/ghgf/tests/data/usdchf.dat")
        input_data = jnp.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        # Start from this source of randomness.
        # We will split keys for subsequent operations.
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model)
        num_samples = 1000
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
        mcmc.run(rng_key_, input_data=input_data)

        samples_1 = mcmc.get_samples(group_by_chain=True)

        assert -1.0 < samples_1["omega_2"].mean() < 0.8


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
