# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict

import jax
import jax.numpy as jnp
from jax.lax import cond

from pyhgf.typing import Edges
from pyhgf.updates.continuous import blank_update
from pyhgf.updates.prediction.binary import predict_binary_state_node
from pyhgf.updates.prediction_error.binary import (
    prediction_error_input_value_parent,
    prediction_error_value_parent,
)


def binary_node_prediction_error(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the value parent(s) of a binary node.

    In a three-level HGF, this step will update the node :math:`x_2`.

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        Interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated node structure.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # retrieve the parent node idx, assuming a unique value parent
    value_parent_idx = jnp.where(
        edges["value_parents"][node_idx], jnp.arange(edges["value_parents"].shape[1]), 0
    ).sum()

    # retrieve the last child index to check before updating
    last_child_idx = jnp.where(
        edges["value_children"][value_parent_idx],
        jnp.arange(edges["value_children"].shape[1]),
        0,
    ).max()

    # if this child is the last one relative to this parent's family, all the
    # children will update the parent at once, otherwise just pass and wait
    attributes = cond(
        node_idx == last_child_idx,
        prediction_error_value_parent,
        blank_update,
        attributes,
        edges,
        value_parent_idx,
    )

    return attributes


def binary_node_prediction(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the expected mean and precision of a binary state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        Interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The new node structure with updated mean and expected precision.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the new expected value for the mean and precision
    expected_precision, expected_mean = predict_binary_state_node(
        attributes, edges, time_step, node_idx
    )

    # Update the node's attributes
    attributes["expected_precision"] = (
        attributes["expected_precision"].at[node_idx].set(expected_precision)
    )
    attributes["expected_mean"] = (
        attributes["expected_mean"].at[node_idx].set(expected_mean)
    )

    return attributes


def binary_input_prediction_error(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Update the input node structure given one binary observation.

    This function is the entry-level of the binary node. It updates the parents of
    the input node (:math:`x_1`).

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
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.

    Returns
    -------
    attributes :
        The updated parameters structure.

    See Also
    --------
    update_continuous_parents, update_continuous_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes["value"] = attributes["value"].at[node_idx].set(value)
    attributes["time_step"] = attributes["time_step"].at[node_idx].set(time_step)

    ####################################################
    # Update the value parent of the binary input node #
    ####################################################

    # retrieve the parent node idx, assuming a unique value parent
    value_parent_idx = jnp.where(
        edges["value_parents"][node_idx], jnp.arange(edges["value_parents"].shape[1]), 0
    ).sum()

    (
        precision_value_parent,
        mean_value_parent,
        surprise,
    ) = prediction_error_input_value_parent(attributes, edges, value_parent_idx)

    jax.debug.print("precision_value_parent: {x} ", x=precision_value_parent)

    # Update value parent's parameters
    attributes["precision"] = (
        attributes["precision"].at[value_parent_idx].set(precision_value_parent)
    )
    attributes["mean"] = attributes["mean"].at[value_parent_idx].set(mean_value_parent)

    # Store surprise in this node
    attributes["surprise"] = attributes["surprise"].at[node_idx].set(surprise)

    return attributes
