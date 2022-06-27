# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Callable, Tuple

import aesara.tensor as at
import jax.numpy as jnp
import numpy as np
from aesara.graph import Apply, Op
from jax import grad, jit
from jax.tree_util import Partial

from ghgf.model import HGF
from ghgf.response import gaussian_surprise


def hgf_logp(
    omega_1: float,
    omega_2: float,
    omega_input: float,
    rho_1: float,
    rho_2: float,
    pi_1: float,
    pi_2: float,
    mu_1: float,
    mu_2: float,
    kappa_1: float = 1.0,
    bias: float = 0.0,
    data: np.ndarray = np.array(0.0),
    response_function: Callable = gaussian_surprise,
    response_function_parameters: Tuple = (),
    model_type: str = "continous",
    n_levels: int = 2,
) -> jnp.DeviceArray:
    """Compute the log probability from the HGF model given the data and parameters.

    Parameters
    ----------
    omega_1, omega_2, omega_3, rho_1, rho_2, rho_3, pi_1, pi_2, pi_3, mu_1, mu_2,
    mu_3, kappa_1, kappa_2, bias : DeviceArray
        The HGD parameters (see py:class:`ghgf.model.HGF` for details).
    data : np.ndarray
        The input data. If `model_type` is `"continuous"`, the data should be two times
        series (time and values).
    response_function : callable
        The response function to use to compute the model surprise.
    response_function_parameters : tuple
        (Optional) Additional parameters to the response function.    model_type : str
    The type of HGF (can be `"continuous"` or `"binary"`).
    model_type : str
        The model type to use (can be "continuous" or "binary").
    n_levels : int
        The number of hierarchies in the perceptual model (can be `2` or `3`). If
        `None`, the nodes hierarchy is not created and might be provided afterward
        using `add_nodes()`. Default sets to `2`.

    """

    # Format HGF parameters
    initial_mu = {"1": mu_1, "2": mu_2}
    initial_pi = {"1": pi_1, "2": pi_2}
    omega = {"1": omega_1, "2": omega_2}
    rho = {"1": rho_1, "2": rho_2}
    kappas = {"1": kappa_1}

    surprise = (
        HGF(
            model_type=model_type,
            initial_mu=initial_mu,
            initial_pi=initial_pi,
            omega=omega,
            omega_input=omega_input,
            rho=rho,
            kappas=kappas,
            bias=bias,
            verbose=False,
            n_levels=n_levels,
        )
        .input_data(data)
        .surprise(
            response_function=response_function,
            response_function_parameters=response_function_parameters,
        )
    )

    # Return the sum of the log probabilities (negative surprise)
    return -surprise


class HGFLogpGradOp(Op):
    def __init__(
        self,
        data: np.ndarray = np.array(0.0),
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Tuple = (),
    ):

        self.grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    n_levels=n_levels,
                    data=data,
                    response_function=response_function,
                    model_type=model_type,
                    response_function_parameters=response_function_parameters,
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            )
        )

    def make_node(
        self,
        omega_1=jnp.array(-3.0),
        omega_2=jnp.array(-3.0),
        omega_input=np.log(1e-4),
        rho_1=jnp.array(0.0),
        rho_2=jnp.array(0.0),
        pi_1=jnp.array(1e4),
        pi_2=jnp.array(1e1),
        mu_1=jnp.array(0.0),
        mu_2=jnp.array(0.0),
        kappa_1=jnp.array(1.0),
        bias=jnp.array(0.0),
    ):

        # Convert our inputs to symbolic variables
        inputs = [
            at.as_tensor_variable(omega_1),
            at.as_tensor_variable(omega_2),
            at.as_tensor_variable(omega_input),
            at.as_tensor_variable(rho_1),
            at.as_tensor_variable(rho_2),
            at.as_tensor_variable(pi_1),
            at.as_tensor_variable(pi_2),
            at.as_tensor_variable(mu_1),
            at.as_tensor_variable(mu_2),
            at.as_tensor_variable(kappa_1),
            at.as_tensor_variable(bias),
        ]
        # This `Op` will return one gradient per input. For simplicity, we assume
        # each output is of the same type as the input. In practice, you should use
        # the exact dtype to avoid overhead when saving the results of the computation
        # in `perform`
        outputs = [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        (
            grad_omega_1,
            grad_omega_2,
            grad_omega_input,
            grad_rho_1,
            grad_rho_2,
            grad_pi_1,
            grad_pi_2,
            grad_mu_1,
            grad_mu_2,
            grad_kappa_1,
            grad_bias,
        ) = self.grad_logp(*inputs)

        outputs[0][0] = np.asarray(grad_omega_1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_omega_2, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(grad_omega_input, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(grad_rho_1, dtype=node.outputs[3].dtype)
        outputs[4][0] = np.asarray(grad_rho_2, dtype=node.outputs[4].dtype)
        outputs[5][0] = np.asarray(grad_pi_1, dtype=node.outputs[5].dtype)
        outputs[6][0] = np.asarray(grad_pi_2, dtype=node.outputs[6].dtype)
        outputs[7][0] = np.asarray(grad_mu_1, dtype=node.outputs[7].dtype)
        outputs[8][0] = np.asarray(grad_mu_2, dtype=node.outputs[8].dtype)
        outputs[9][0] = np.asarray(grad_kappa_1, dtype=node.outputs[9].dtype)
        outputs[10][0] = np.asarray(grad_bias, dtype=node.outputs[10].dtype)


class HGFDistribution(Op):
    """The HGF distribution to be used in the context of a PyMC 4.0 model.

    Examples
    --------
    This class should be used in the context of a PyMC probabilistic model to estimate
    one or many parameter-s probability density with MCMC sampling.

    .. python::

        import arviz as az
        import numpy as np
        import pymc as pm
        from math import log
        from ghgf import load_data
        from ghgf.pymc import HGFDistribution
        from ghgf.response import gaussian_surprise

        # Create the data (value and time vectors)
        timeserie = load_data("continuous")
        input_data = np.array(
            [timeserie, np.arange(1, len(timeserie) + 1, dtype=float)]
        ).T

        # We create the PyMC distribution here, specifying the type of model and
        # response function we want to use (i.e simple gaussian surprise).
        hgf_logp_op = HGFDistribution(
            n_levels=2,
            data=input_data,
            response_function=gaussian_surprise,
        )

        # Create the PyMC model
        # Here we are fixing all the paramters to arbitrary values and sampling the
        # posterior probability density of Omega in the second level, using a normal
        # distribution as prior : ω_2 ~ N(-11, 2).

        with pm.Model() as model:

            ω_2 = pm.Normal("ω_2", -11.0, 2)

            pm.Potential(
                "hhgf_loglike",
                hgf_logp_op(
                    omega_1=0.0,
                    omega_2=ω_2,
                    omega_input=log(1e-4),
                    rho_1=0.0,
                    rho_2=0.0,
                    pi_1=1e4,
                    pi_2=1e1,
                    mu_1=input_data[0, 0],
                    mu_2=0.0,
                    kappa_1=1.0,
                    bias=0.0,
                ),
            )

        # Sample the model - Using 4 chain, 4 cores and 1000 warmups.
        with model:
            hgf_samples = pm.sample(chains=4, cores=4, tune=1000)

        # Print a summary of the results using Arviz
        az.summary(hgf_samples, var_names="ω_2")
        ===  =======  =====  =======  =======  =====  =====  ====  ====  =
        ω_2  -12.846  1.305  -15.466  -10.675  0.038  0.027  1254  1726  1
        ===  =======  =====  =======  =======  =====  =====  ====  ====  =

    """

    def __init__(
        self,
        data: np.ndarray = np.array(0.0),
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Tuple = (),
    ):
        """

        Parameters
        ----------

        """
        # The value function
        self.hgf_logp = jit(
            Partial(
                hgf_logp,
                n_levels=n_levels,
                data=data,
                response_function=response_function,
                model_type=model_type,
                response_function_parameters=response_function_parameters,
            ),
        )

        # The gradient function
        self.hgf_logp_grad_op = HGFLogpGradOp(
            data=data,
            model_type=model_type,
            n_levels=n_levels,
            response_function=response_function,
            response_function_parameters=response_function_parameters,
        )

    def make_node(
        self,
        omega_1: float,
        omega_2: float,
        omega_input: float,
        rho_1: float,
        rho_2: float,
        pi_1: float,
        pi_2: float,
        mu_1: float,
        mu_2: float,
        kappa_1: float,
        bias: float,
    ):

        # Convert our inputs to symbolic variables
        inputs = [
            at.as_tensor_variable(omega_1),
            at.as_tensor_variable(omega_2),
            at.as_tensor_variable(omega_input),
            at.as_tensor_variable(rho_1),
            at.as_tensor_variable(rho_2),
            at.as_tensor_variable(pi_1),
            at.as_tensor_variable(pi_2),
            at.as_tensor_variable(mu_1),
            at.as_tensor_variable(mu_2),
            at.as_tensor_variable(kappa_1),
            at.as_tensor_variable(bias),
        ]
        # Define the type of the output returned by the wrapped JAX function
        outputs = [at.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        result = self.hgf_logp(*inputs)
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        (
            grad_omega_1,
            grad_omega_2,
            grad_omega_input,
            grad_rho_1,
            grad_rho_2,
            grad_pi_1,
            grad_pi_2,
            grad_mu_1,
            grad_mu_2,
            grad_kappa_1,
            grad_bias,
        ) = self.hgf_logp_grad_op(*inputs)
        # If there are inputs for which the gradients will never be needed or cannot
        # be computed, `aesara.gradient.grad_not_implemented` should  be used as the
        # output gradient for that input.
        output_gradient = output_gradients[0]
        return [
            output_gradient * grad_omega_1,
            output_gradient * grad_omega_2,
            output_gradient * grad_omega_input,
            output_gradient * grad_rho_1,
            output_gradient * grad_rho_2,
            output_gradient * grad_pi_1,
            output_gradient * grad_pi_2,
            output_gradient * grad_mu_1,
            output_gradient * grad_mu_2,
            output_gradient * grad_kappa_1,
            output_gradient * grad_bias,
        ]
