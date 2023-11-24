# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx"))
def continuous_node_value_prediction_error(
    attributes: Dict,
    node_idx: int,
) -> Array:
    r"""Compute the value prediction error and expected precision of a state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    prediction_error_precision_value_parent, prediction_error_input_mean_value_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # value prediction error
    value_prediction_error = (
        attributes[node_idx]["mean"] - attributes[node_idx]["expected_mean"]
    )

    # send to the value parent node for later use in the update step
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error

    return attributes


@partial(jit, static_argnames=("node_idx"))
def continuous_node_volatility_prediction_error(
    attributes: Dict, node_idx: int
) -> Dict:
    r"""Compute the volatility prediction error of a state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    prediction_error_mean_volatility_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # compute the volatility prediction error (VOPE)
    volatility_prediction_error = (
        (attributes[node_idx]["expected_precision"] / attributes[node_idx]["precision"])
        + attributes[node_idx]["expected_precision"]
        * (attributes[node_idx]["temp"]["value_prediction_error"]) ** 2
        - 1
    )

    attributes[node_idx]["temp"][
        "volatility_prediction_error"
    ] = volatility_prediction_error

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_error(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Store prediction errors in an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    node_idx :
        Pointer to the continuous node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_input_value_prediction_error, continuous_input_value_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Store value prediction errors
    # -----------------------------
    attributes = continuous_node_value_prediction_error(
        attributes=attributes, node_idx=node_idx
    )

    # Store volatility prediction errors
    # ----------------------------------
    attributes = continuous_node_volatility_prediction_error(
        attributes=attributes, node_idx=node_idx
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_blank_update(
    attributes: Dict, edges: Edges, time_step: float, node_idx: int
) -> Array:
    r"""Compute the new precision of a continuous state node node when input is missing.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network that contains the continuous state
        node.
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.
    time_step :
        The time interval between the previous time point and the current time point.
    node_idx :
        Index of the node that should be updated.

    Returns
    -------
    precision :
        The new expected precision of the value parent.

    """
    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling * attributes[volatility_parents_idx]["mean"]
            )

    # compute the predicted_volatility from the total volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)
    predicted_volatility = jnp.where(
        predicted_volatility > 1e-128, predicted_volatility, jnp.nan
    )

    # Estimate the new precision for the continuous state node
    precision = 1 / ((1 / attributes[node_idx]["precision"]) + predicted_volatility)

    return precision
