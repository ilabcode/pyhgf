# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import unittest
from unittest import TestCase

import arviz as az
import jax.numpy as jnp
import numpy as np
import pymc as pm
from jax import grad, jit
from jax.tree_util import Partial

from pyhgf import load_data
from pyhgf.distribution import HGFDistribution, HGFLogpGradOp, hgf_logp
from pyhgf.response import (
    binary_softmax_inverse_temperature,
    first_level_gaussian_surprise,
)


class TestDistribution(TestCase):
    def test_hgf_logp(self):
        ##################
        # Continuous HGF #
        ##################

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")
        jax_logp = jit(
            Partial(
                hgf_logp,
                n_levels=2,
                input_data=[timeserie],
                response_function=first_level_gaussian_surprise,
                model_type="continuous",
                response_function_inputs=None,
                time_steps=None,
            )
        )

        logp = jax_logp(
            tonic_volatility_1=-3.0,
            tonic_volatility_2=-3.0,
            tonic_volatility_3=jnp.nan,
            continuous_precision=np.array(1e4),
            binary_precision=np.inf,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            tonic_drift_3=jnp.nan,
            precision_1=1e4,
            precision_2=1e1,
            precision_3=jnp.nan,
            mean_1=timeserie[0],
            mean_2=0.0,
            mean_3=jnp.nan,
            volatility_coupling_1=1.0,
            volatility_coupling_2=jnp.nan,
            response_function_parameters=[jnp.nan],
        )
        assert jnp.isclose(logp, 1141.0911)

        ##############
        # Binary HGF #
        ##############

        # Create the data (value and time vectors)
        u, y = load_data("binary")
        jax_logp = jit(
            Partial(
                hgf_logp,
                n_levels=2,
                input_data=[u],
                response_function=binary_softmax_inverse_temperature,
                model_type="binary",
                response_function_inputs=[y],
            )
        )

        logp = jax_logp(
            tonic_volatility_1=jnp.inf,
            tonic_volatility_2=jnp.array(-6.0),
            tonic_volatility_3=jnp.nan,
            continuous_precision=jnp.nan,
            binary_precision=jnp.inf,
            tonic_drift_1=jnp.array(0.0),
            tonic_drift_2=jnp.array(0.0),
            tonic_drift_3=jnp.nan,
            precision_1=jnp.array(0.0),
            precision_2=jnp.array(1e4),
            precision_3=jnp.nan,
            mean_1=jnp.inf,
            mean_2=jnp.array(0.5),
            mean_3=jnp.nan,
            volatility_coupling_1=jnp.array(1.0),
            volatility_coupling_2=jnp.nan,
            response_function_parameters=[1.0],
        )
        assert jnp.isclose(logp, -200.24422)

    def test_grad_logp(self):
        ##################
        # Continuous HGF #
        ##################

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")
        grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    n_levels=2,
                    input_data=[timeserie],
                    response_function=first_level_gaussian_surprise,
                    model_type="continuous",
                    response_function_inputs=None,
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ),
        )

        (
            tonic_volatility_1,
            tonic_volatility_2,
            tonic_volatility_3,
            continuous_precision,
            binary_precision,
            tonic_drift_1,
            tonic_drift_2,
            tonic_drift_3,
            precision_1,
            precision_2,
            precision_3,
            mean_1,
            mean_2,
            mean_3,
            volatility_coupling_1,
            volatility_coupling_2,
            response_function_parameters,
        ) = grad_logp(
            np.array(-3.0),
            np.array(-3.0),
            np.array(0.0),
            np.array(1e4),
            np.nan,
            np.array(0.0),
            np.array(0.0),
            np.array(0.0),
            np.array(1e4),
            np.array(1e1),
            np.array(0.0),
            np.array(timeserie[0]),
            np.array(0.0),
            np.array(0.0),
            np.array(1.0),
            np.array(0.0),
            np.array([np.nan]),
        )

        assert jnp.isclose(tonic_volatility_1, -7.864815)

        ##############
        # Binary HGF #
        ##############

        # Create the data (value and time vectors)
        u, y = load_data("binary")

        grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    n_levels=2,
                    input_data=[u],
                    response_function=binary_softmax_inverse_temperature,
                    model_type="binary",
                    response_function_inputs=[y],
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ),
        )

        (
            tonic_volatility_1,
            tonic_volatility_2,
            tonic_volatility_3,
            continuous_precision,
            binary_precision,
            tonic_drift_1,
            tonic_drift_2,
            tonic_drift_3,
            precision_1,
            precision_2,
            precision_3,
            mean_1,
            mean_2,
            mean_3,
            volatility_coupling_1,
            volatility_coupling_2,
            response_function_parameters,
        ) = grad_logp(
            np.array(0.0),
            np.array(-2.0),
            np.array(0.0),
            np.array(1e4),
            np.inf,
            np.array(0.0),
            np.array(0.0),
            np.array(0.0),
            np.array(0.0),
            np.array(1e4),
            np.array(0.0),
            np.array(0.0),
            np.array(0.5),
            np.array(0.0),
            np.array(1.0),
            np.array(1.0),
            np.array([1.0]),
        )

        assert jnp.isclose(tonic_volatility_2, 15.896532)

    def test_aesara_logp(self):
        """Test the aesara hgf_logp op."""

        ##################
        # Continuous HGF #
        ##################

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf_logp_op = HGFDistribution(
            input_data=[timeserie],
            model_type="continuous",
            n_levels=2,
            response_function=first_level_gaussian_surprise,
            response_function_inputs=None,
        )

        logp = hgf_logp_op(
            tonic_volatility_1=np.array(-3.0),
            tonic_volatility_2=np.array(-3.0),
            tonic_volatility_3=np.array(0.0),
            continuous_precision=np.array(1e4),
            binary_precision=np.inf,
            tonic_drift_1=np.array(0.0),
            tonic_drift_2=np.array(0.0),
            tonic_drift_3=np.array(0.0),
            precision_1=np.array(1e4),
            precision_2=np.array(1e1),
            precision_3=np.array(0.0),
            mean_1=np.array(timeserie[0]),
            mean_2=np.array(0.0),
            mean_3=np.array(0.0),
            volatility_coupling_1=np.array(1.0),
            volatility_coupling_2=np.array(0.0),
            response_function_parameters=np.array([np.nan]),
        ).eval()

        assert jnp.isclose(logp, 1141.09106445)

        ##############
        # Binary HGF #
        ##############

        # Create the data (value and time vectors)
        u, y = load_data("binary")

        hgf_logp_op = HGFDistribution(
            input_data=[u],
            model_type="binary",
            n_levels=2,
            response_function=binary_softmax_inverse_temperature,
            response_function_inputs=[y],
        )

        logp = hgf_logp_op(
            tonic_volatility_1=np.inf,
            tonic_volatility_2=-6.0,
            tonic_volatility_3=np.inf,
            continuous_precision=np.array(1e4),
            binary_precision=np.inf,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            tonic_drift_3=np.inf,
            precision_1=0.0,
            precision_2=1e4,
            precision_3=np.inf,
            mean_1=np.inf,
            mean_2=0.5,
            mean_3=np.inf,
            volatility_coupling_1=1.0,
            volatility_coupling_2=np.inf,
            response_function_parameters=np.array([1.0]),
        ).eval()

        assert jnp.isclose(logp, -200.24421692)

    def test_aesara_grad_logp(self):
        """Test the aesara gradient hgf_logp op."""

        ##################
        # Continuous HGF #
        ##################

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf_logp_grad_op = HGFLogpGradOp(
            model_type="continuous",
            input_data=[timeserie],
            n_levels=2,
            response_function=first_level_gaussian_surprise,
            response_function_inputs=None,
        )

        tonic_volatility_1 = hgf_logp_grad_op(
            tonic_volatility_1=-3.0,
            tonic_volatility_2=-3.0,
            continuous_precision=np.array(1e4),
            binary_precision=np.nan,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            precision_1=1e4,
            precision_2=1e1,
            mean_1=timeserie[0],
            mean_2=0.0,
            volatility_coupling_1=1.0,
        )[0].eval()

        assert jnp.isclose(tonic_volatility_1, -7.864815)

        ##############
        # Binary HGF #
        ##############

        # Create the data (value and time vectors)
        u, y = load_data("binary")

        hgf_logp_grad_op = HGFLogpGradOp(
            model_type="binary",
            input_data=[u],
            n_levels=2,
            response_function=binary_softmax_inverse_temperature,
            response_function_inputs=[y],
        )

        tonic_volatility_2 = hgf_logp_grad_op(
            tonic_volatility_1=jnp.inf,
            tonic_volatility_2=-6.0,
            continuous_precision=jnp.nan,
            binary_precision=jnp.inf,
            tonic_drift_1=0.0,
            tonic_drift_2=0.0,
            precision_1=0.0,
            precision_2=1e4,
            mean_1=jnp.inf,
            mean_2=0.5,
            volatility_coupling_1=1.0,
        )[1].eval()

        assert jnp.isclose(tonic_volatility_2, 10.8664665)

    def test_pymc_sampling(self):
        """Test the aesara hgf_logp op."""

        ##############
        # Continuous #
        ##############

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")

        hgf_logp_op = HGFDistribution(
            n_levels=2,
            input_data=[timeserie],
            response_function=first_level_gaussian_surprise,
        )

        with pm.Model() as model:
            tonic_volatility_2 = pm.Uniform("tonic_volatility_2", -10, 0)

            pm.Potential(
                "hhgf_loglike",
                hgf_logp_op(
                    tonic_volatility_2=tonic_volatility_2,
                    continuous_precision=np.array(1e4),
                ),
            )

        initial_point = model.initial_point()

        pointslogs = model.point_logps(initial_point)
        assert pointslogs["tonic_volatility_2"] == -1.39
        assert pointslogs["hhgf_loglike"] == 1468.28

        with model:
            idata = pm.sample(chains=2, cores=1, tune=1000)

        assert -9.5 > az.summary(idata)["mean"].values[0] > -10.5
        assert az.summary(idata)["r_hat"].values[0] <= 1.02

        ##########
        # Binary #
        ##########

        # Create the data (value and time vectors)
        u, y = load_data("binary")

        hgf_logp_op = HGFDistribution(
            n_levels=2,
            model_type="binary",
            input_data=[u],
            response_function=binary_softmax_inverse_temperature,
            response_function_inputs=[y],
        )

        with pm.Model() as model:
            tonic_volatility_2 = pm.Normal("tonic_volatility_2", -11.0, 2)

            pm.Potential(
                "hhgf_loglike",
                hgf_logp_op(
                    tonic_volatility_2=tonic_volatility_2,
                    response_function_parameters=[1.0],
                ),
            )

        initial_point = model.initial_point()

        pointslogs = model.point_logps(initial_point)
        assert pointslogs["tonic_volatility_2"] == -1.61
        assert pointslogs["hhgf_loglike"] == -212.59

        with model:
            idata = pm.sample(chains=2, cores=1, tune=1000)

        assert -2 < round(az.summary(idata)["mean"].values[0]) < 0


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
