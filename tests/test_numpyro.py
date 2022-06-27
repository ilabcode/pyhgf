# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import arviz as az
import numpy as np
import numpyro as npy
import numpyro.distributions as dist
from jax import random
from numpyro.infer import MCMC, NUTS

from ghgf import load_data
from ghgf.numpyro import HGFDistribution


class Testnumpyro(TestCase):
    def test_HGFDistribution(self):
        """Test the Numpyro distribution"""

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, np.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        omega_1, omega_2 = -3.0, -3.0
        rho_1, rho_2 = 0.0, 0.0
        pi_1, pi_2 = 1e4, 1e1
        mu_1, mu_2 = input_data[0, 0], 0.0
        kappa = 1.0
        bias = 0.0

        hgf = HGFDistribution(
            input_data=input_data,
            omega_1=omega_1,
            omega_2=omega_2,
            rho_1=rho_1,
            rho_2=rho_2,
            pi_1=pi_1,
            pi_2=pi_2,
            mu_1=mu_1,
            mu_2=mu_2,
            kappa_1=kappa,
            bias=bias,
        )

        assert np.isclose(hgf.log_prob(None), 1938.0101)

        #################
        # Test sampling #
        #################

        def model(input_data):

            ω_1 = 0.0
            ω_2 = npy.sample("ω_2", dist.Normal(-11.0, 2.0))
            rho_1 = 0.0
            rho_2 = 0.0
            pi_1 = 1e4
            pi_2 = 1e1
            μ_1 = input_data[0, 0]
            μ_2 = 0.0
            kappa_1 = 1.0

            npy.sample(
                "hgf_log_prob",
                HGFDistribution(
                    input_data=input_data,
                    omega_1=ω_1,
                    omega_2=ω_2,
                    rho_1=rho_1,
                    rho_2=rho_2,
                    pi_1=pi_1,
                    pi_2=pi_2,
                    mu_1=μ_1,
                    mu_2=μ_2,
                    kappa_1=kappa_1,
                ),
            )

        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        # Run NUTS.
        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        mcmc.run(rng_key_, input_data=input_data)

        numpyro_samples = az.from_numpyro(mcmc)
        assert (
            -14
            < round(az.summary(numpyro_samples, var_names="ω_2")["mean"].values[0])
            < -10
        )


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
