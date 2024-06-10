# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_update_infinite(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a binary node given infinite precision of the input.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    value_child_idx = edges[node_idx].value_children[0]  # type: ignore

    attributes[node_idx]["mean"] = attributes[value_child_idx]["values"]
    attributes[node_idx]["precision"] = attributes[value_child_idx][
        "expected_precision"
    ]
    attributes[node_idx]["observed"] = attributes[value_child_idx]["observed"]

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_update_finite(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a binary node given finite precision of the input.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    value_child_idx = edges[node_idx].value_children[0]  # type: ignore

    delata0 = attributes[value_child_idx]["temp"]["value_prediction_error_0"]
    delata1 = attributes[value_child_idx]["temp"]["value_prediction_error_1"]
    expected_precision = attributes[value_child_idx]["expected_precision"]

    # Likelihood under eta1
    und1 = jnp.exp(-expected_precision / 2 * delata1**2)

    # Likelihood under eta0
    und0 = jnp.exp(-expected_precision / 2 * delata0**2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    expected_mean = attributes[node_idx]["expected_mean"]
    mean = expected_mean * und1 / (expected_mean * und1 + (1 - expected_mean) * und0)
    precision = 1 / (expected_mean * (1 - expected_mean))

    attributes[node_idx]["mean"] = mean
    attributes[node_idx]["precision"] = precision

    return attributes
