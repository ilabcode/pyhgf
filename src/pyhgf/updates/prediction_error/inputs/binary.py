# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction_error_infinite_precision(
    attributes: Dict, node_idx: int, time_step: float, value: float
) -> Dict:
    r"""Compute the prediction error of a binary input assuming an infinite precision.

    Parameters
    ----------
    value :
        The new observed value.
    time_step :
        The interval between the previous time point and the current time point.
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction_error_finite_precision(
    attributes: Dict, edges: Edges, node_idx: int, time_step: float, value: float
) -> Dict:
    r"""Compute the prediction error of a binary input assuming a finite precision.

    Parameters
    ----------
    value :
        The new observed value.
    time_step :
        The interval between the previous time point and the current time point.
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step

    # Read parameters from the binary input
    eta0 = attributes[node_idx]["eta0"]
    eta1 = attributes[node_idx]["eta1"]

    # compute two prediction errors given the two possible values
    attributes[node_idx]["temp"]["value_prediction_error_0"] = value - eta0
    attributes[node_idx]["temp"]["value_prediction_error_1"] = value - eta1

    return attributes
