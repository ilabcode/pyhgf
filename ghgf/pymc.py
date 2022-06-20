# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Optional

import aesara.tensor as at
import jax.numpy as jnp
import numpy as np
from aesara.graph import Apply, Op
from jax import grad, jit
from jax.interpreters.xla import DeviceArray
from jax.lax import scan

from ghgf.hgf_jax import loop_inputs
from ghgf.model import HGF


@jit
def hgf_logp(
    data: DeviceArray,
    omega_1: Optional[DeviceArray] = None,
    omega_2: Optional[DeviceArray] = None,
    omega_input: DeviceArray = jnp.log(1e-4),
    rho_1: Optional[DeviceArray] = None,
    rho_2: Optional[DeviceArray] = None,
    pi_1: Optional[DeviceArray] = None,
    pi_2: Optional[DeviceArray] = None,
    mu_1: Optional[DeviceArray] = None,
    mu_2: Optional[DeviceArray] = None,
    kappa_1: DeviceArray = jnp.array(1.0),
    bias: DeviceArray = jnp.array(0.0),
) -> jnp.DeviceArray:
    """Compute the log probability from the HGF model given the data and parameters."""

    # Transpose data if time is not the first dimension
    if (data.shape[0] == 2) & (data.shape[1] > 2):
        data = data.T

    # Format HGF parameters
    initial_mu = {"1": mu_1, "2": mu_2}
    initial_pi = {"1": pi_1, "2": pi_2}
    omega = {"1": omega_1, "2": omega_2}
    rho = {"1": rho_1, "2": rho_2}
    kappas = {"1": kappa_1}

    hgf = HGF(
        initial_mu=initial_mu,
        initial_pi=initial_pi,
        omega=omega,
        omega_input=omega_input,
        rho=rho,
        kappas=kappas,
        bias=bias,
        verbose=False,
    )

    # Create the input structure
    # Here we use the first row from the input data
    res_init = (
        hgf.input_node,
        {
            "time": data[0, 1],
            "value": data[0, 0] + bias,
            "surprise": jnp.array(0.0),
        },
    )

    # This is where the HGF functions is used to scan the input time series
    _, final = scan(loop_inputs, res_init, data[1:, :])
    _, results = final

    # Fill surprises with zeros if invalid input
    this_surprise = jnp.where(
        jnp.any(jnp.isnan(data[1:, :]), axis=1), 0.0, results["surprise"]
    )
    # Return an infinite surprise if the model cannot fit
    this_surprise = jnp.where(jnp.isnan(this_surprise), jnp.inf, this_surprise)

    # Sum the surprise for this model
    this_logp = jnp.sum(this_surprise)

    # Return the negative of the sum of the log probabilities
    return -this_logp


grad_hgf_logp = jit(grad(hgf_logp, argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))


class HGFLogpGradOp(Op):
    def make_node(
        self,
        data: DeviceArray,
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
            at.as_tensor_variable(data),
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
            grad_input,
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
        ) = grad_hgf_logp(*inputs)

        outputs[0][0] = np.asarray(grad_input, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_omega_1, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(grad_omega_2, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(grad_omega_input, dtype=node.outputs[3].dtype)
        outputs[4][0] = np.asarray(grad_rho_1, dtype=node.outputs[4].dtype)
        outputs[5][0] = np.asarray(grad_rho_2, dtype=node.outputs[5].dtype)
        outputs[6][0] = np.asarray(grad_pi_1, dtype=node.outputs[6].dtype)
        outputs[7][0] = np.asarray(grad_pi_2, dtype=node.outputs[7].dtype)
        outputs[8][0] = np.asarray(grad_mu_1, dtype=node.outputs[8].dtype)
        outputs[9][0] = np.asarray(grad_mu_2, dtype=node.outputs[9].dtype)
        outputs[10][0] = np.asarray(grad_kappa_1, dtype=node.outputs[10].dtype)
        outputs[11][0] = np.asarray(grad_bias, dtype=node.outputs[11].dtype)


# Initialize our `Op`s
hgf_logp_grad_op = HGFLogpGradOp()


class HGFLogpOp(Op):
    def make_node(
        self,
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
    ):
        # Convert our inputs to symbolic variables
        inputs = [
            at.as_tensor_variable(data),
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
        result = hgf_logp(*inputs)
        # Aesara raises an error if the dtype of the returned output is not
        # exactly the one expected from the Apply node (in this case
        # `dscalar`, which stands for float64 scalar), so we make sure
        # to convert to the expected dtype. To avoid unnecessary conversions
        # you should make sure the expected output defined in `make_node`
        # is already of the correct dtype
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        (
            grad_input,
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
        ) = hgf_logp_grad_op(*inputs)
        # If there are inputs for which the gradients will never be needed or cannot
        # be computed, `aesara.gradient.grad_not_implemented` should  be used as the
        # output gradient for that input.
        output_gradient = output_gradients[0]
        return [
            output_gradient * grad_input,
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


hgf_logp_op = HGFLogpOp()
