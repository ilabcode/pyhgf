# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.math import binary_surprise
from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx", "edges"))
def binary_input_prediction_error_infinite_precision(
    attributes: Dict,
    time_step: float,
    edges: Edges,
    node_idx: int,
    value: float,
    observed: bool,
    **args
) -> Dict:
    """Compute the prediction error of a binary input assuming an infinite precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    value :
        The new observed value.
    observed :
        Whether value was observed or not.

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
    attributes[node_idx]["values"] = value
    attributes[node_idx]["observed"] = observed
    attributes[node_idx]["time_step"] = time_step

    value_parent_idx = edges[node_idx].value_parents[0]  # type: ignore
    attributes[node_idx]["surprise"] = (
        binary_surprise(
            x=value, expected_mean=attributes[value_parent_idx]["expected_mean"]
        )
        * observed
    )

    return attributes


@partial(jit, static_argnames=("node_idx"))
def binary_input_prediction_error_finite_precision(
    attributes: Dict, time_step: float, node_idx: int, value: float, **args
) -> Dict:
    r"""Compute the prediction error of a binary input assuming a finite precision.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the value parent node that will be updated.
    value :
        The new observed value.

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
    attributes[node_idx]["values"] = value
    attributes[node_idx]["time_step"] = time_step

    # Read parameters from the binary input
    eta0 = attributes[node_idx]["eta0"]
    eta1 = attributes[node_idx]["eta1"]

    # compute two prediction errors given the two possible values
    attributes[node_idx]["temp"]["value_prediction_error_0"] = value - eta0
    attributes[node_idx]["temp"]["value_prediction_error_1"] = value - eta1

    return attributes
