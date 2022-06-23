# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import os
import unittest
from unittest import TestCase

import arviz as az
import jax.numpy as jnp
import numpy as np
import pymc as pm
from numpy import loadtxt

from ghgf.pymc import HGFLogpGradOp, HGFLogpOp, grad_hgf_logp, hgf_logp


class Testsdt(TestCase):
    def test_hgf_logp(self):

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        logp = hgf_logp(
            data=input_data,
            omega_1=jnp.array(-3.0),
            omega_2=jnp.array(-3.0),
            omega_input=jnp.log(1e-4),
            rho_1=jnp.array(0.0),
            rho_2=jnp.array(0.0),
            pi_1=jnp.array(1e4),
            pi_2=jnp.array(1e1),
            mu_1=jnp.array(input_data[0][0]),
            mu_2=jnp.array(0.0),
            kappa_1=jnp.array(1.0),
            bias=jnp.array(0.0),
            model_type="continuous",
            n_levels=2,
            response_function="GaussianSurprise",
            response_function_parameters=(1, 1),
        )
        assert jnp.isclose(logp, 1938.0101)

    def test_grad_logp(self):

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        (
            data,
            omega_1,
            omega_2,
            omega_input,
            rho_1,
            rho_2,
            pi_1,
            pi_2,
            mu_1,
            mu_2,
            kappa_1,
            bias,
            response_function_parameters,
        ) = grad_hgf_logp(
            input_data,
            np.array(-3.0),
            np.array(-3.0),
            np.log(1e-4),
            np.array(0.0),
            np.array(0.0),
            np.array(1e4),
            np.array(1e1),
            np.array(input_data[0][0]),
            np.array(0.0),
            np.array(1.0),
            np.array(0.0),
            (1, 1),
            model_type="continuous",
            n_levels=2,
            response_function="GaussianSurprise",
        )

        assert jnp.isclose(omega_1, 0.47931308)
        # assert jnp.isclose(omega_2, 1938.0101)
        # assert jnp.isclose(omega_input, 1938.0101)
        # assert jnp.isclose(rho_1, 1938.0101)
        # assert jnp.isclose(rho_2, 1938.0101)
        # assert jnp.isclose(pi_1, 1938.0101)
        # assert jnp.isclose(pi_2, 1938.0101)
        # assert jnp.isclose(mu_1, 1938.0101)
        # assert jnp.isclose(mu_2, 1938.0101)
        # assert jnp.isclose(kappa_1, 1938.0101)
        # assert jnp.isclose(bias, 1938.0101)

    def test_aesara_logp(self):
        """Test the aesara hgf_logp op."""

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        hgf_logp_op = HGFLogpOp(
            model_type="continuous",
            n_levels=2,
            response_function="GaussianSurprise",
            response_function_parameters=(1, 1),
        )

        logp = hgf_logp_op(
            data=input_data,
            omega_1=np.array(-3.0),
            omega_2=np.array(-3.0),
            omega_input=np.log(1e-4),
            rho_1=np.array(0.0),
            rho_2=np.array(0.0),
            pi_1=np.array(1e4),
            pi_2=np.array(1e1),
            mu_1=np.array(input_data[0][0]),
            mu_2=np.array(0.0),
            kappa_1=np.array(1.0),
            bias=np.array(0.0),
        ).eval()

        assert jnp.isclose(logp, 1938.01013184)

    def test_aesara_grad_logp(self):
        """Test the aesara gradient hgf_logp op."""

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        hgf_logp_grad_op = HGFLogpGradOp(
            model_type="continuous",
            n_levels=2,
            response_function="GaussianSurprise",
            response_function_parameters=(1, 1),
        )

        omega_1 = hgf_logp_grad_op(
            data=input_data,
            omega_1=np.array(-3.0),
            omega_2=np.array(-3.0),
            omega_input=np.log(1e-4),
            rho_1=np.array(0.0),
            rho_2=np.array(0.0),
            pi_1=np.array(1e4),
            pi_2=np.array(1e1),
            mu_1=np.array(input_data[0][0]),
            mu_2=np.array(0.0),
            kappa_1=np.array(1.0),
            bias=np.array(0.0),
        )[1].eval()

        assert jnp.isclose(omega_1, 0.47931308)

    def test_pymc_sampling(self):
        """Test the aesara hgf_logp op."""

        # Create the data (value and time vectors)
        timeserie = loadtxt(os.path.dirname(__file__) + "/data/usdchf.dat")

        # Repeate the input time series 3 time to test for multiple models
        input_data = np.array(
            [timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        hgf_logp_op = HGFLogpOp(
            n_levels=2,
            response_function="GaussianSurprise",
            response_function_parameters=(np.array(1), 1),
        )

        with pm.Model() as model:

            # omega_1 = pm.Normal("omega_1", -3.0, 2)
            omega_2 = pm.Normal("omega_2", -11.0, 2)

            pm.Potential(
                "hhgf_loglike",
                hgf_logp_op(
                    data=input_data,
                    omega_1=np.array(0.0),
                    omega_2=omega_2,
                    omega_input=np.log(1e-4),
                    rho_1=np.array(0.0),
                    rho_2=np.array(0.0),
                    pi_1=np.array(1e4),
                    pi_2=np.array(1e1),
                    mu_1=np.array(input_data[0][0]),
                    mu_2=np.array(0.0),
                    kappa_1=np.array(1.0),
                    bias=np.array(0.0),
                ),
            )

        initial_point = model.initial_point()

        pointslogs = model.point_logps(initial_point)
        assert pointslogs["omega_2"] == -1.61
        assert pointslogs["hhgf_loglike"] == 2149.04

        with model:
            idata = pm.sample(chains=4, cores=4, tune=10000, target_accept=0.95)

        assert -14 < round(az.summary(idata)["mean"].values[0]) < -10
        # az.plot_trace(idata)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
