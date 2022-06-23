# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Callable, Tuple

import aesara.tensor as at
import jax.numpy as jnp
import numpy as np
from aesara.graph import Apply, Op
from jax import grad, jit
from jax.interpreters.xla import DeviceArray
from jax.tree_util import Partial

from ghgf.model import HGF
from ghgf.response import gaussian_surprise


def hgf_logp(
    omega_1: DeviceArray = jnp.array(-3.0),
    omega_2: DeviceArray = jnp.array(-3.0),
    omega_input: DeviceArray = jnp.log(1e-4),
    rho_1: DeviceArray = jnp.array(0.0),
    rho_2: DeviceArray = jnp.array(0.0),
    pi_1: DeviceArray = jnp.array(1e4),
    pi_2: DeviceArray = jnp.array(1e1),
    mu_1: DeviceArray = jnp.array(0.0),
    mu_2: DeviceArray = jnp.array(0.0),
    kappa_1: DeviceArray = jnp.array(1.0),
    bias: DeviceArray = jnp.array(0.0),
    data: np.ndarray = np.array(0.0),
    response_function_parameters: Tuple = (),
    model_type: str = "continous",
    n_levels: int = 2,
    response_function: Callable = gaussian_surprise,
) -> jnp.DeviceArray:
    """Compute the log probability from the HGF model given the data and parameters.

    Parameters
    ----------
    data : DeviceArray
        The input data. If `model_type` is `"continuous"`, the data should be two times
        series (time and values).
    ** HGF parameters (see ghgf.model.HGF).
    model_type : str
        The type of HGF (can be `"continuous"` or `"binary"`).
    n_levels : int
        The number of hierarchies in the perceptual model (can be `2` or `3`). If
        `None`, the nodes hierarchy is not created and might be provided afterward
        using `add_nodes()`. Default sets to `2`.
    response_function : Callable
        The response function used to compute the log probability. Defaults to
        py:func`ghgf.response.gaussiangurprise`.
    response_function_parameters : tuple
        Dictionary containing additional data that will be passed to the custom
        response function.

    """

    # Format HGF parameters
    initial_mu = {"1": mu_1, "2": mu_2}
    initial_pi = {"1": pi_1, "2": pi_2}
    omega = {"1": omega_1, "2": omega_2}
    rho = {"1": rho_1, "2": rho_2}
    kappas = {"1": kappa_1}

    hgf = HGF(
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

    # Create the input structure - use the first row from the input data
    # This is where the HGF functions is used to scan the input time series
    hgf.input_data(data)

    hgf_results = hgf.hgf_results

    # Return the model evidence
    logp = response_function(
        hgf_results=hgf_results,
        response_function_parameters=response_function_parameters,
    )

    # Return the negative of the sum of the log probabilities
    return -logp


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


class HGFLogpOp(Op):
    def __init__(
        self,
        data: np.ndarray = np.array(0.0),
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Tuple = (),
    ):
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
