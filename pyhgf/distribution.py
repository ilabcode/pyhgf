# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from jax import grad, jit
from jax.tree_util import Partial
from pytensor.graph import Apply, Op

from pyhgf.model import HGF


def hgf_logp(
    omega_1: float,
    omega_2: float,
    omega_3: float,
    omega_input: float,
    rho_1: float,
    rho_2: float,
    rho_3: float,
    pi_1: float,
    pi_2: float,
    pi_3: float,
    mu_1: float,
    mu_2: float,
    mu_3: float,
    kappa_1: float,
    kappa_2: float,
    input_data: List[np.ndarray],
    response_function: Callable,
    model_type: str,
    n_levels: int,
    response_function_parameters: Tuple = (),
    time: Optional[List] = None,
) -> jnp.DeviceArray:
    r"""Log probability from HGF model(s) given input data and parameter(s).

    This function support broadcasting along the first axis:

    - If the input data contains many time series, the function will automatically
    create the corresponding number of HGF models and fit them separately.
    - If a single input data is provided but some parameters have array-like inputs, the
    number of HGF models will match the length of the arrays, using the value *i*
    for the *i* th model. When floats are provided for some parameters, the same value
    will be used for all HGF models.
    - If multiple input data are provided with array-like inputs for some parameter, the
    function will create and fit the models separately using the value *i* for the
    *i* th model.

    Parameters
    ----------
    omega_1 : float
        The :math:`\omega` parameter, or *evolution rate*, at the first level of the
        HGF. This parameter represents the tonic part of the variance (the part that is
        not inherited from parents nodes).
    omega_2 : float
        The :math:`\omega` parameter, or *evolution rate*, at the second level of the
        HGF. This parameter represents the tonic part of the variance (the part that is
        not inherited from parents nodes).
    omega_3 : float
        The :math:`\omega` parameter, or *evolution rate*, at the third level of the
        HGF. This parameter represents the tonic part of the variance (the part that is
        not inherited from parents nodes). The value of this parameter will be ignored
        when using a 2-levels HGF (`n_levels=2`).
    omega_input : float
        Represent the noise associated with the input observation.
    rho_1 : float
        The :math:`\rho` parameter at the first level of the HGF. This parameter
        represents the drift of the random walk.
    rho_2 : float
        The :math:`\rho` parameter at the second level of the HGF. This parameter
        represents the drift of the random walk.
    rho_3 : float
        The :math:`\rho` parameter at the first level of the HGF. This parameter
        represents the drift of the random walk. The value of this parameter will be
        ignored when using a 2-levels HGF (`n_levels=2`).
    pi_1 : float
        The :math:`\pi` parameter, or *precision*, at the first level of the HGF.
    pi_2 : float
        The :math:`\pi` parameter, or *precision*, at the second level of the HGF.
    pi_3 : float
        The :math:`\pi` parameter, or *precision*, at the third level of the HGF. The
        value of this parameter will be ignored when using a 2-levels HGF
        (`n_levels=2`).
    mu_1 : float
        The :math:`\mu` parameter, or *mean*, at the first level of the HGF.
    mu_2 : float
        The :math:`\mu` parameter, or *mean*, at the second level of the HGF.
    mu_3 : float
        The :math:`\mu` parameter, or *mean*, at the third level of the HGF. The value
        of this parameter will be ignored when using a 2-levels HGF (`n_levels=2`).
    kappa_1 : float
        The value of the :math:`\\kappa` parameter at the first level of the HGF. Kappa
        represents the phasic part of the variance (the part that is affected by the
        parents nodes) and will defines the strenght of the connection between the node
        and the parent node. Often fixed to `1`.
    kappa_2 : float
        The value of the :math:`\\kappa` parameter at the second level of the HGF. Kappa
        represents the phasic part of the variance (the part that is affected by the
        parents nodes) and will defines the strenght of the connection between the node
        and the parent node. Often fixed to `1`. The value of this parameter will be
        ignored when using a 2-levels HGF (`n_levels=2`).
    input_data : list
        List of input data. When `n` models should be fitted, the list contains `n` 1d
        Numpy arrays. By default, the associated time vector is the integers vector
        starting at `0`. A different time vector can be passed to the `time` argument.
    response_function : callable
        The response function to use to compute the model surprise.
    model_type : str
        The model type to use (can be "continuous" or "binary").
    n_levels : int
        The number of hierarchies in the perceptual model (can be `2` or `3`). If
        `None`, the nodes hierarchy is not created and might be provided afterward
        using :py:meth:`pyhgf.model.HGF.add_nodes`.
    response_function_parameters : tuple
        Additional parameters to the response function.
    time : list | optional
        List of 1d Numpy arrays containing the time vectors for each input time series.
        If one of the list item is `None`, or if `None` is provided instead, the time
        vector will defaults to integers vector starting at 0.

    Returns
    -------
    log_prob : DeviceArray
        The sum of the log-probabilities (or negative surprise).

    """
    # number of models
    n = len(input_data)

    # Broadcast inputs to an array with length n>=1
    (
        omega_1,
        omega_2,
        omega_3,
        omega_input,
        rho_1,
        rho_2,
        rho_3,
        pi_1,
        pi_2,
        pi_3,
        mu_1,
        mu_2,
        mu_3,
        kappa_1,
        kappa_2,
        _,
    ) = jnp.broadcast_arrays(
        omega_1,
        omega_2,
        omega_3,
        omega_input,
        rho_1,
        rho_2,
        rho_3,
        pi_1,
        pi_2,
        pi_3,
        mu_1,
        mu_2,
        mu_3,
        kappa_1,
        kappa_2,
        jnp.zeros(n),
    )

    # if no time vectors provided, set it to None (will defaults to integers vectors)
    # otherwise check consistency with the input data
    if time is None:
        time = [None] * n
    else:
        assert len(time) == n

    surprise = 0.0

    # Fitting n HGF models to the n datasets
    for i in range(n):

        # Format HGF parameters
        initial_mu: Dict[str, Optional[float]] = {
            "1": mu_1[i],
            "2": mu_2[i],
            "3": mu_3[i],
        }
        initial_pi: Dict[str, Optional[float]] = {
            "1": pi_1[i],
            "2": pi_2[i],
            "3": pi_3[i],
        }
        omega: Dict[str, Optional[float]] = {
            "1": omega_1[i],
            "2": omega_2[i],
            "3": omega_3[i],
        }
        rho: Dict[str, Optional[float]] = {"1": rho_1[i], "2": rho_2[i], "3": rho_3[i]}
        kappas: Dict[str, Optional[float]] = {"1": kappa_1[i], "2": kappa_2[i]}

        surprise = surprise + (
            HGF(
                initial_mu=initial_mu,
                initial_pi=initial_pi,
                omega=omega,
                omega_input=omega_input[i],
                rho=rho,
                kappas=kappas,
                model_type=model_type,
                n_levels=n_levels,
                eta0=0.0,
                eta1=1.0,
                pihat=jnp.inf,
                verbose=False,
            )
            .input_data(input_data=input_data[i], time=time[i])
            .surprise(
                response_function=response_function,
                response_function_parameters=response_function_parameters,
            )
        )

    # Return the sum of the log probabilities (negative surprise)
    return -surprise


class HGFLogpGradOp(Op):
    """Gradient Op for the HGF distribution."""

    def __init__(
        self,
        input_data: List = [],
        time: Optional[List] = None,
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_parameters: Tuple = (),
    ):
        """Initialize function.

        Parameters
        ----------
        input_data : list
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time vector is the integers
            vector starting at `0`. A different time vector can be passed to the `time`
            argument.
        time : list | optional
            List of 1d Numpy arrays containing the time vectors for each input time
            series. If one of the list item is `None`, or if `None` is provided instead,
            the time vector will defaults to integers vector starting at 0.
        model_type : str
            The model type to use (can be "continuous" or "binary").
        n_levels : int
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`.
        response_function : callable
            The response function to use to compute the model surprise.
        response_function_parameters : tuple
            Additional parameters to the response function.

        """
        self.grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    n_levels=n_levels,
                    input_data=input_data,
                    time=time,
                    response_function=response_function,
                    model_type=model_type,
                    response_function_parameters=response_function_parameters,
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            )
        )

    def make_node(
        self,
        omega_1=np.array(-3.0),
        omega_2=np.array(-3.0),
        omega_3=np.array(0.0),
        omega_input=np.log(1e-4),
        rho_1=np.array(0.0),
        rho_2=np.array(0.0),
        rho_3=np.array(0.0),
        pi_1=np.array(1e4),
        pi_2=np.array(1e1),
        pi_3=np.array(0.0),
        mu_1=np.array(0.0),
        mu_2=np.array(0.0),
        mu_3=np.array(0.0),
        kappa_1=np.array(1.0),
        kappa_2=np.array(0.0),
    ):
        """Initialize node structure."""
        # Convert our inputs to symbolic variables
        inputs = [
            pt.as_tensor_variable(omega_1),
            pt.as_tensor_variable(omega_2),
            pt.as_tensor_variable(omega_3),
            pt.as_tensor_variable(omega_input),
            pt.as_tensor_variable(rho_1),
            pt.as_tensor_variable(rho_2),
            pt.as_tensor_variable(rho_3),
            pt.as_tensor_variable(pi_1),
            pt.as_tensor_variable(pi_2),
            pt.as_tensor_variable(pi_3),
            pt.as_tensor_variable(mu_1),
            pt.as_tensor_variable(mu_2),
            pt.as_tensor_variable(mu_3),
            pt.as_tensor_variable(kappa_1),
            pt.as_tensor_variable(kappa_2),
        ]
        # This `Op` will return one gradient per input. For simplicity, we assume
        # each output is of the same type as the input. In practice, you should use
        # the exact dtype to avoid overhead when saving the results of the computation
        # in `perform`
        outputs = [inp.type() for inp in inputs]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Perform node operations."""
        (
            grad_omega_1,
            grad_omega_2,
            grad_omega_3,
            grad_omega_input,
            grad_rho_1,
            grad_rho_2,
            grad_rho_3,
            grad_pi_1,
            grad_pi_2,
            grad_pi_3,
            grad_mu_1,
            grad_mu_2,
            grad_mu_3,
            grad_kappa_1,
            grad_kappa_2,
        ) = self.grad_logp(*inputs)

        outputs[0][0] = np.asarray(grad_omega_1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_omega_2, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(grad_omega_3, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(grad_omega_input, dtype=node.outputs[3].dtype)
        outputs[4][0] = np.asarray(grad_rho_1, dtype=node.outputs[4].dtype)
        outputs[5][0] = np.asarray(grad_rho_2, dtype=node.outputs[5].dtype)
        outputs[6][0] = np.asarray(grad_rho_3, dtype=node.outputs[6].dtype)
        outputs[7][0] = np.asarray(grad_pi_1, dtype=node.outputs[7].dtype)
        outputs[8][0] = np.asarray(grad_pi_2, dtype=node.outputs[8].dtype)
        outputs[9][0] = np.asarray(grad_pi_3, dtype=node.outputs[9].dtype)
        outputs[10][0] = np.asarray(grad_mu_1, dtype=node.outputs[10].dtype)
        outputs[11][0] = np.asarray(grad_mu_2, dtype=node.outputs[11].dtype)
        outputs[12][0] = np.asarray(grad_mu_3, dtype=node.outputs[12].dtype)
        outputs[13][0] = np.asarray(grad_kappa_1, dtype=node.outputs[13].dtype)
        outputs[14][0] = np.asarray(grad_kappa_2, dtype=node.outputs[14].dtype)


class HGFDistribution(Op):
    """The HGF distribution PyMC >= 5.0 compatible.

    Examples
    --------
    This class should be used in the context of a PyMC probabilistic model to estimate
    one or many parameter-s probability density with MCMC sampling.

    >>> import arviz as az
    >>> import numpy as np
    >>> import pymc as pm
    >>> from math import log
    >>> from pyhgf import load_data
    >>> from pyhgf.distribution import HGFDistribution
    >>> from pyhgf.response import total_gaussian_surprise

    Create the data (value and time vectors)

    >>> timeserie = load_data("continuous")

    We create the PyMC distribution here, specifying the type of model and response
    function we want to use (i.e simple gaussian surprise).

    >>> hgf_logp_op = HGFDistribution(
    >>>     n_levels=2,
    >>>     input_data=timeserie,
    >>>     response_function=total_gaussian_surprise,
    >>> )

    Create the PyMC model
    Here we are fixing all the paramters to arbitrary values and sampling the
    posterior probability density of Omega in the second level, using a normal
    distribution as prior : ω_2 ~ N(-11, 2).

    >>> with pm.Model() as model:
    >>>     ω_2 = pm.Normal("ω_2", -11.0, 2)
    >>>     pm.Potential(
    >>>         "hhgf_loglike",
    >>>         hgf_logp_op(
    >>>             omega_1=0.0,
    >>>             omega_2=ω_2,
    >>>             omega_input=log(1e-4),
    >>>             rho_1=0.0,
    >>>             rho_2=0.0,
    >>>             pi_1=1e4,
    >>>             pi_2=1e1,
    >>>             mu_1=timeserie[0],
    >>>             mu_2=0.0,
    >>>             kappa_1=1.0,
    >>>         ),
    >>>     )

    Sample the model - Using 4 chain, 4 cores and 1000 warmups.

    >>> with model:
    >>>     hgf_samples = pm.sample(chains=4, cores=4, tune=1000)

    Print a summary of the results using Arviz

    >>> az.summary(hgf_samples, var_names="ω_2")
    ===  =======  =====  =======  =======  =====  =====  ====  ====  =
    ω_2  -12.846  1.305  -15.466  -10.675  0.038  0.027  1254  1726  1
    ===  =======  =====  =======  =======  =====  =====  ====  ====  =

    """

    def __init__(
        self,
        input_data: List[np.array] = [],
        time: Optional[List] = None,
        model_type: str = "continuous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_parameters: Tuple = (),
    ):
        """Distribution initialization.

        Parameters
        ----------
        input_data : list
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time vector is the integers
            vector starting at `0`. A different time vector can be passed to the `time`
            argument.
        time : list | optional
            List of 1d Numpy arrays containing the time vectors for each input time
            series. If one of the list item is `None`, or if `None` is provided instead,
            the time vector will defaults to integers vector starting at 0.
        model_type : str
            The model type to use (can be "continuous" or "binary").
        n_levels : int
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`.
        response_function : callable
            The response function to use to compute the model surprise.
        response_function_parameters : tuple
            Additional parameters to the response function.

        """
        # The value function
        self.hgf_logp = jit(
            Partial(
                hgf_logp,
                n_levels=n_levels,
                input_data=input_data,
                time=time,
                response_function=response_function,
                model_type=model_type,
                response_function_parameters=response_function_parameters,
            ),
        )

        # The gradient function
        self.hgf_logp_grad_op = HGFLogpGradOp(
            input_data=input_data,
            time=time,
            model_type=model_type,
            n_levels=n_levels,
            response_function=response_function,
            response_function_parameters=response_function_parameters,
        )

    def make_node(
        self,
        omega_1,
        omega_2,
        omega_3,
        omega_input,
        rho_1,
        rho_2,
        rho_3,
        pi_1,
        pi_2,
        pi_3,
        mu_1,
        mu_2,
        mu_3,
        kappa_1,
        kappa_2,
    ):
        """Convert inputs to symbolic variables."""
        inputs = [
            pt.as_tensor_variable(omega_1),
            pt.as_tensor_variable(omega_2),
            pt.as_tensor_variable(omega_3),
            pt.as_tensor_variable(omega_input),
            pt.as_tensor_variable(rho_1),
            pt.as_tensor_variable(rho_2),
            pt.as_tensor_variable(rho_3),
            pt.as_tensor_variable(pi_1),
            pt.as_tensor_variable(pi_2),
            pt.as_tensor_variable(pi_3),
            pt.as_tensor_variable(mu_1),
            pt.as_tensor_variable(mu_2),
            pt.as_tensor_variable(mu_3),
            pt.as_tensor_variable(kappa_1),
            pt.as_tensor_variable(kappa_2),
        ]
        # Define the type of the output returned by the wrapped JAX function
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Run the function forward."""
        result = self.hgf_logp(*inputs)
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        """Gradient of the function."""
        (
            grad_omega_1,
            grad_omega_2,
            grad_omega_3,
            grad_omega_input,
            grad_rho_1,
            grad_rho_2,
            grad_rho_3,
            grad_pi_1,
            grad_pi_2,
            grad_pi_3,
            grad_mu_1,
            grad_mu_2,
            grad_mu_3,
            grad_kappa_1,
            grad_kappa_2,
        ) = self.hgf_logp_grad_op(*inputs)
        # If there are inputs for which the gradients will never be needed or cannot
        # be computed, `aesara.gradient.grad_not_implemented` should  be used as the
        # output gradient for that input.
        output_gradient = output_gradients[0]
        return [
            output_gradient * grad_omega_1,
            output_gradient * grad_omega_2,
            output_gradient * grad_omega_3,
            output_gradient * grad_omega_input,
            output_gradient * grad_rho_1,
            output_gradient * grad_rho_2,
            output_gradient * grad_rho_3,
            output_gradient * grad_pi_1,
            output_gradient * grad_pi_2,
            output_gradient * grad_pi_3,
            output_gradient * grad_mu_1,
            output_gradient * grad_mu_2,
            output_gradient * grad_mu_3,
            output_gradient * grad_kappa_1,
            output_gradient * grad_kappa_2,
        ]
