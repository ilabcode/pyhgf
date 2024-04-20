# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pytensor.tensor as pt
from jax import grad, jit
from jax.tree_util import Partial
from jax.typing import ArrayLike
from pytensor.graph import Apply, Op

from pyhgf.model import HGF


def hgf_logp(
    tonic_volatility_1: Union[np.ndarray, ArrayLike, float] = -3.0,
    tonic_volatility_2: Union[np.ndarray, ArrayLike, float] = -3.0,
    tonic_volatility_3: Union[np.ndarray, ArrayLike, float] = -3.0,
    continuous_precision: Union[np.ndarray, ArrayLike, float] = 1e4,
    binary_precision: Union[np.ndarray, ArrayLike, float] = np.inf,
    tonic_drift_1: Union[np.ndarray, ArrayLike, float] = 0.0,
    tonic_drift_2: Union[np.ndarray, ArrayLike, float] = 0.0,
    tonic_drift_3: Union[np.ndarray, ArrayLike, float] = 0.0,
    precision_1: Union[np.ndarray, ArrayLike, float] = 1.0,
    precision_2: Union[np.ndarray, ArrayLike, float] = 1.0,
    precision_3: Union[np.ndarray, ArrayLike, float] = 1.0,
    mean_1: Union[np.ndarray, ArrayLike, float] = 0.0,
    mean_2: Union[np.ndarray, ArrayLike, float] = 0.0,
    mean_3: Union[np.ndarray, ArrayLike, float] = 0.0,
    volatility_coupling_1: Union[np.ndarray, ArrayLike, float] = 1.0,
    volatility_coupling_2: Union[np.ndarray, ArrayLike, float] = 1.0,
    response_function_parameters: List[Union[np.ndarray, ArrayLike, float]] = [1.0],
    input_data: List[np.ndarray] = [np.nan],
    response_function: Optional[Callable] = None,
    model_type: str = "continuous",
    n_levels: int = 2,
    response_function_inputs: List[np.ndarray] = [np.nan],
    time_steps: Optional[List] = None,
) -> float:
    r"""HGF log-probability given input data, response function and parameters.

    .. note::

       This function support broadcasting along the first axis, which means that it can
       fit multiple HGF when multiple data points are provided:
       * If the input data contains many time series, the function will automatically
       create the corresponding number of HGF models and fit them separately.
       * If a single input data is provided but some parameters have array-like inputs,
       the number of HGF models will match the length of the arrays, using the value
       *i* for the *i* th model. When floats are provided for some parameters, the same
       value will be used for all HGF models.
       * If multiple input data are provided with array-like inputs for some parameter,
       the function will create and fit the models separately using the value *i* for
       the *i* th model.

    Parameters
    ----------
    tonic_volatility_1 :
        The tonic volatility at the first level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parents
        nodes).
    tonic_volatility_2 :
        The tonic volatility at the second level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parents
        nodes).
    tonic_volatility_3 :
        The tonic volatility at the third level of the HGF. This parameter represents
        the tonic part of the variance (the part that is not inherited from parents
        nodes). The value of this parameter will be ignored when using a two-level HGF
        (`n_levels=2`).
    continuous_precision :
        The expected precision associated with the continuous input.
    binary_precision :
        The expected precision associated with the binary input.
    tonic_drift_1 :
        The tonic drift at the first level of the HGF. This parameter represents the
        drift of the random walk.
    tonic_drift_2 :
        The tonic drift at the second level of the HGF. This parameter represents the
        drift of the random walk.
    tonic_drift_3 :
        The tonic drift at the first level of the HGF. This parameter represents the
        drift of the random walk. The value of this parameter will be ignored when
        using a two-level HGF (`n_levels=2`).
    precision_1 :
        The precision at the first level of the HGF.
    precision_2 :
        The precision at the second level of the HGF.
    precision_3 :
        The precision at the third level of the HGF. The value of this parameter will
        be ignored when using a two-level HGF (`n_levels=2`).
    mean_1 :
        The mean at the first level of the HGF.
    mean_2 :
        The mean at the second level of the HGF.
    mean_3 :
        The mean at the third level of the HGF. The value of this parameter will be
        ignored when using a two-level HGF (`n_levels=2`).
    volatility_coupling_1 :
        The volatility coupling between the first and second level of the HGF. This
        represents the phasic part of the variance (the part that is affected by the
        parent nodes). Defaults to `1.0`.
    volatility_coupling_2 :
        The volatility coupling between the second and third level of the HGF. This
        represents the phasic part of the variance (the part that is affected by the
        parent nodes). Defaults to `1.0`. The value of this parameter will be
        ignored when using a two-level HGF (`n_levels=2`).
    input_data :
        List of input data. When `n` models should be fitted, the list contains `n` 1d
        Numpy arrays. By default, the associated time vector is the integers vector
        starting at `0`. A different time vector can be passed to the `time` argument.
    response_function :
        The response function to use to compute the model surprise.
    model_type :
        The model type to use (can be "continuous" or "binary").
    n_levels :
        The number of hierarchies in the perceptual model (can be `2` or `3`). If
        `None`, the nodes hierarchy is not created and might be provided afterwards
        using :py:meth:`pyhgf.model.HGF.add_nodes`.
    response_function_inputs :
        A list of tuples with the same length as the number of models. Each tuple
        contains additional data and parameters that can be accessible to the response
        functions.
    response_function_parameters :
        A list of additional parameters that will be passed to the response function.
        This can include values over which inferece is performed in a PyMC model (e.g.
        the inverse temperature of a binary softmax).
    time_steps :
        List of 1d Numpy arrays containing the time vectors for each input time series.
        If one of the list items is `None`, or if `None` is provided instead, the time
        vector will default to the unit vector.

    Returns
    -------
    log_prob :
        The sum of the log probabilities (or negative surprise).

    """
    # number of models
    n = len(input_data)

    # Broadcast inputs to an array with length n>=1
    (
        _tonic_volatility_1,
        _tonic_volatility_2,
        _tonic_volatility_3,
        _continuous_precision,
        _binary_precision,
        _tonic_drift_1,
        _tonic_drift_2,
        _tonic_drift_3,
        _precision_1,
        _precision_2,
        _precision_3,
        _mean_1,
        _mean_2,
        _mean_3,
        _volatility_coupling_1,
        _volatility_coupling_2,
        _,
    ) = jnp.broadcast_arrays(
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
        jnp.zeros(n),
    )

    if response_function_inputs is None:
        response_function_inputs = [None] * n
    if response_function_parameters is None:
        response_function_parameters = [None] * n

    # if no time vectors provided, set it to None (will defaults to integers vectors)
    # otherwise check consistency with the input data
    if time_steps is None:
        time_steps = [None] * n
    else:
        assert len(time_steps) == n

    surprise = 0.0

    # Fitting n HGF models to the n datasets
    for i in range(n):
        # Format HGF parameters
        initial_mean: Dict = {
            "1": _mean_1[i],
            "2": _mean_2[i],
            "3": _mean_3[i],
        }
        initial_precision: Dict = {
            "1": _precision_1[i],
            "2": _precision_2[i],
            "3": _precision_3[i],
        }
        tonic_volatility: Dict = {
            "1": _tonic_volatility_1[i],
            "2": _tonic_volatility_2[i],
            "3": _tonic_volatility_3[i],
        }
        tonic_drift: Dict = {
            "1": _tonic_drift_1[i],
            "2": _tonic_drift_2[i],
            "3": _tonic_drift_3[i],
        }
        volatility_coupling: Dict = {
            "1": _volatility_coupling_1[i],
            "2": _volatility_coupling_2[i],
        }

        surprise = surprise + (
            HGF(
                initial_mean=initial_mean,
                initial_precision=initial_precision,
                tonic_volatility=tonic_volatility,
                continuous_precision=_continuous_precision[i],
                binary_precision=_binary_precision[i],
                tonic_drift=tonic_drift,
                volatility_coupling=volatility_coupling,
                model_type=model_type,
                n_levels=n_levels,
                eta0=0.0,
                eta1=1.0,
                verbose=False,
            )
            .input_data(input_data=input_data[i], time_steps=time_steps[i])
            .surprise(
                response_function=response_function,
                response_function_inputs=response_function_inputs[i],
                response_function_parameters=response_function_parameters[i],
            )
        )

    # Return the sum of the log probabilities (negative surprise)
    return -surprise


class HGFLogpGradOp(Op):
    """Gradient Op for the HGF distribution."""

    def __init__(
        self,
        input_data: List = [],
        time_steps: Optional[List] = None,
        model_type: str = "continous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[List[Optional[Tuple]]] = None,
    ):
        """Initialize function.

        Parameters
        ----------
        input_data :
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time_steps vector is the unit
            vector. A different time vector can be passed to the `time_steps` argument.
        time_steps :
            List of 1d Numpy arrays containing the time_steps vectors for each input
            time series. If one of the list items is `None`, or if `None` is provided
            instead, the time_steps vector will default to an integers vector starting
            at 0.
        model_type :
            The model type to use (can be "continuous" or "binary").
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`.
        response_function :
            The response function to use to compute the model surprise.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.

        """
        self.grad_logp = jit(
            grad(
                Partial(
                    hgf_logp,
                    n_levels=n_levels,
                    input_data=input_data,
                    time_steps=time_steps,
                    response_function=response_function,
                    model_type=model_type,
                    response_function_inputs=response_function_inputs,
                ),
                argnums=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            )
        )

    def make_node(
        self,
        tonic_volatility_1=np.array(-3.0),
        tonic_volatility_2=np.array(-3.0),
        tonic_volatility_3=np.array(-3.0),
        continuous_precision=np.array(1e4),
        binary_precision=np.inf,
        tonic_drift_1=np.array(0.0),
        tonic_drift_2=np.array(0.0),
        tonic_drift_3=np.array(0.0),
        precision_1=np.array(1.0),
        precision_2=np.array(1.0),
        precision_3=np.array(0.0),
        mean_1=np.array(0.0),
        mean_2=np.array(0.0),
        mean_3=np.array(0.0),
        volatility_coupling_1=np.array(1.0),
        volatility_coupling_2=np.array(1.0),
        response_function_parameters=np.array([1.0]),
    ):
        """Initialize node structure."""
        # Convert our inputs to symbolic variables
        inputs = [
            pt.as_tensor_variable(tonic_volatility_1),
            pt.as_tensor_variable(tonic_volatility_2),
            pt.as_tensor_variable(tonic_volatility_3),
            pt.as_tensor_variable(continuous_precision),
            pt.as_tensor_variable(binary_precision),
            pt.as_tensor_variable(tonic_drift_1),
            pt.as_tensor_variable(tonic_drift_2),
            pt.as_tensor_variable(tonic_drift_3),
            pt.as_tensor_variable(precision_1),
            pt.as_tensor_variable(precision_2),
            pt.as_tensor_variable(precision_3),
            pt.as_tensor_variable(mean_1),
            pt.as_tensor_variable(mean_2),
            pt.as_tensor_variable(mean_3),
            pt.as_tensor_variable(volatility_coupling_1),
            pt.as_tensor_variable(volatility_coupling_2),
            pt.as_tensor_variable(response_function_parameters),
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
            grad_tonic_volatility_1,
            grad_tonic_volatility_2,
            grad_tonic_volatility_3,
            grad_continuous_precision,
            grad_binary_precision,
            grad_tonic_drift_1,
            grad_tonic_drift_2,
            grad_tonic_drift_3,
            grad_precision_1,
            grad_precision_2,
            grad_precision_3,
            grad_mean_1,
            grad_mean_2,
            grad_mean_3,
            grad_volatility_coupling_1,
            grad_volatility_coupling_2,
            grad_response_function_parameters,
        ) = self.grad_logp(*inputs)

        outputs[0][0] = np.asarray(grad_tonic_volatility_1, dtype=node.outputs[0].dtype)
        outputs[1][0] = np.asarray(grad_tonic_volatility_2, dtype=node.outputs[1].dtype)
        outputs[2][0] = np.asarray(grad_tonic_volatility_3, dtype=node.outputs[2].dtype)
        outputs[3][0] = np.asarray(
            grad_continuous_precision, dtype=node.outputs[3].dtype
        )
        outputs[4][0] = np.asarray(grad_binary_precision, dtype=node.outputs[4].dtype)
        outputs[5][0] = np.asarray(grad_tonic_drift_1, dtype=node.outputs[5].dtype)
        outputs[6][0] = np.asarray(grad_tonic_drift_2, dtype=node.outputs[6].dtype)
        outputs[7][0] = np.asarray(grad_tonic_drift_3, dtype=node.outputs[7].dtype)
        outputs[8][0] = np.asarray(grad_precision_1, dtype=node.outputs[8].dtype)
        outputs[9][0] = np.asarray(grad_precision_2, dtype=node.outputs[9].dtype)
        outputs[10][0] = np.asarray(grad_precision_3, dtype=node.outputs[10].dtype)
        outputs[11][0] = np.asarray(grad_mean_1, dtype=node.outputs[11].dtype)
        outputs[12][0] = np.asarray(grad_mean_2, dtype=node.outputs[12].dtype)
        outputs[13][0] = np.asarray(grad_mean_3, dtype=node.outputs[13].dtype)
        outputs[14][0] = np.asarray(
            grad_volatility_coupling_1, dtype=node.outputs[14].dtype
        )
        outputs[15][0] = np.asarray(
            grad_volatility_coupling_2, dtype=node.outputs[15].dtype
        )
        outputs[16][0] = np.asarray(
            grad_response_function_parameters, dtype=node.outputs[16].dtype
        )


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
    function we want to use (i.e. simple Gaussian surprise).

    >>> hgf_logp_op = HGFDistribution(
    >>>     n_levels=2,
    >>>     input_data=timeserie,
    >>>     response_function=total_gaussian_surprise,
    >>> )

    Create the PyMC model
    Here we are fixing all the parameters to arbitrary values and sampling the
    posterior probability density of Omega in the second level, using a normal
    distribution as prior : ω_2 ~ N(-11, 2).

    >>> with pm.Model() as model:
    >>>     ω_2 = pm.Normal("ω_2", -11.0, 2)
    >>>     pm.Potential(
    >>>         "hhgf_loglike",
    >>>         hgf_logp_op(
    >>>             tonic_volatility_2=ω_2,
    >>>             precision_1=1e4,
    >>>             precision_2=1e1,
    >>>             mean_1=timeserie[0],
    >>>         ),
    >>>     )

    Sample the model - Using 4 chain, 4 cores and 1000 warmups.

    >>> with model:
    >>>     hgf_samples = pm.sample(chains=2, cores=1, tune=1000)

    Print a summary of the results using Arviz

    >>> az.summary(hgf_samples, var_names="ω_2")
    ===  =======  =====  =======  =======  =====  =====  ====  ====  =
    ω_2  -12.846  1.305  -15.466  -10.675  0.038  0.027  1254  1726  1
    ===  =======  =====  =======  =======  =====  =====  ====  ====  =

    """

    def __init__(
        self,
        input_data: List[np.ndarray] = [],
        time_steps: Optional[List] = None,
        model_type: str = "continuous",
        n_levels: int = 2,
        response_function: Optional[Callable] = None,
        response_function_inputs: Optional[List[Optional[Tuple]]] = None,
    ):
        """Distribution initialization.

        Parameters
        ----------
        input_data :
            List of input data. When `n` models should be fitted, the list contains `n`
            1d Numpy arrays. By default, the associated time vector is the unit
            vector starting at `0`. A different time_steps vector can be passed to
            the `time_steps` argument.
        time_steps :
            List of 1d Numpy arrays containing the time_steps vectors for each input
            time series. If one of the list items is `None`, or if `None` is provided
            instead, the time vector will default to an integers vector starting at 0.
        model_type :
            The model type to use (can be "continuous" or "binary").
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards
            using `add_nodes()`.
        response_function :
            The response function to use to compute the model surprise.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.

        """
        # The value function
        self.hgf_logp = jit(
            Partial(
                hgf_logp,
                n_levels=n_levels,
                input_data=input_data,
                time_steps=time_steps,
                response_function=response_function,
                model_type=model_type,
                response_function_inputs=response_function_inputs,
            ),
        )

        # The gradient function
        self.hgf_logp_grad_op = HGFLogpGradOp(
            input_data=input_data,
            time_steps=time_steps,
            model_type=model_type,
            n_levels=n_levels,
            response_function=response_function,
            response_function_inputs=response_function_inputs,
        )

    def make_node(
        self,
        tonic_volatility_1=np.array(-3.0),
        tonic_volatility_2=np.array(-3.0),
        tonic_volatility_3=np.array(-3.0),
        continuous_precision=np.array(1e4),
        binary_precision=np.inf,
        tonic_drift_1=np.array(0.0),
        tonic_drift_2=np.array(0.0),
        tonic_drift_3=np.array(0.0),
        precision_1=np.array(1.0),
        precision_2=np.array(1.0),
        precision_3=np.array(1.0),
        mean_1=np.array(0.0),
        mean_2=np.array(0.0),
        mean_3=np.array(0.0),
        volatility_coupling_1=np.array(1.0),
        volatility_coupling_2=np.array(1.0),
        response_function_parameters=np.array([1.0]),
    ):
        """Convert inputs to symbolic variables."""
        inputs = [
            pt.as_tensor_variable(tonic_volatility_1),
            pt.as_tensor_variable(tonic_volatility_2),
            pt.as_tensor_variable(tonic_volatility_3),
            pt.as_tensor_variable(continuous_precision),
            pt.as_tensor_variable(binary_precision),
            pt.as_tensor_variable(tonic_drift_1),
            pt.as_tensor_variable(tonic_drift_2),
            pt.as_tensor_variable(tonic_drift_3),
            pt.as_tensor_variable(precision_1),
            pt.as_tensor_variable(precision_2),
            pt.as_tensor_variable(precision_3),
            pt.as_tensor_variable(mean_1),
            pt.as_tensor_variable(mean_2),
            pt.as_tensor_variable(mean_3),
            pt.as_tensor_variable(volatility_coupling_1),
            pt.as_tensor_variable(volatility_coupling_2),
            pt.as_tensor_variable(response_function_parameters),
        ]
        # Define the type of output returned by the wrapped JAX function
        outputs = [pt.dscalar()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Run the function forward."""
        result = self.hgf_logp(*inputs)
        outputs[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        """Gradient of the function."""
        (
            grad_tonic_volatility_1,
            grad_tonic_volatility_2,
            grad_tonic_volatility_3,
            grad_continuous_precision,
            grad_binary_precision,
            grad_tonic_drift_1,
            grad_tonic_drift_2,
            grad_tonic_drift_3,
            grad_precision_1,
            grad_precision_2,
            grad_precision_3,
            grad_mean_1,
            grad_mean_2,
            grad_mean_3,
            grad_volatility_coupling_1,
            grad_volatility_coupling_2,
            grad_response_function_parameters,
        ) = self.hgf_logp_grad_op(*inputs)
        # If there are inputs for which the gradients will never be needed or cannot
        # be computed, `pytensor.gradient.grad_not_implemented` should  be used as the
        # output gradient for that input.
        output_gradient = output_gradients[0]
        return [
            output_gradient * grad_tonic_volatility_1,
            output_gradient * grad_tonic_volatility_2,
            output_gradient * grad_tonic_volatility_3,
            output_gradient * grad_continuous_precision,
            output_gradient * grad_binary_precision,
            output_gradient * grad_tonic_drift_1,
            output_gradient * grad_tonic_drift_2,
            output_gradient * grad_tonic_drift_3,
            output_gradient * grad_precision_1,
            output_gradient * grad_precision_2,
            output_gradient * grad_precision_3,
            output_gradient * grad_mean_1,
            output_gradient * grad_mean_2,
            output_gradient * grad_mean_3,
            output_gradient * grad_volatility_coupling_1,
            output_gradient * grad_volatility_coupling_2,
            output_gradient * grad_response_function_parameters,
        ]
