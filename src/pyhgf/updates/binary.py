# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from jax.tree_util import Partial

from pyhgf.typing import Edges
from pyhgf.updates.prediction.binary import predict_binary_state_node
from pyhgf.updates.prediction_error.inputs.binary import (
    prediction_error_input_value_parent,
)
from pyhgf.updates.prediction_error.nodes.binary import prediction_error_value_parent
from pyhgf.updates.prediction_error.nodes.continuous import continuous_blank_update


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_blank_update(
    attributes: Dict,
    time_step: float,
    value: float,
    node_idx: int,
    edges: Edges,
) -> Dict:
    """Update of a binary state node in the case of missing observation."""
    # using the current node index, unwrap parameters and parents
    value_parent_idxs = edges[node_idx].value_parents

    # Return here if no parent nodes are found
    if value_parent_idxs is None:
        return attributes

    # ----------------------------------------------------------------------------------
    # Update the value parent of the binary input node - When the binary input is
    # missing, we only update the continuous parent node of the binary state node using
    # the blank update, which will later influence the binary state predictions.
    # ----------------------------------------------------------------------------------
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # estimate the new precision as a function of time elapsed
            precision_value_parent = continuous_blank_update(
                attributes, edges, time_step, value_parent_idx
            )

            # Update the continuous parent's precision
            attributes[value_parent_idx]["precision"] = precision_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_prediction_error_missing_input(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the value parent(s) of a binary state node in case of missing input.

    In a three-level HGF, this step will update the node :math:`x_2`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility of parents and
        children.

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
    blank_fn = Partial(binary_node_blank_update, node_idx=node_idx, edges=edges)

    regular_fn = Partial(binary_node_prediction_error, node_idx=node_idx, edges=edges)

    # retrieve the value from the binary input
    # assuming univariate descendency for now
    input_value = attributes[edges[node_idx].value_children[0]]["value"]  # type: ignore

    # if the observation is missing, use the blank update
    # otherwise use the regular update for binary inputs
    attributes = cond(
        jnp.isnan(input_value), blank_fn, regular_fn, attributes, time_step
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_prediction_error(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the value parent(s) of a binary state node.

    In a three-level HGF, this step will update the node :math:`x_2`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility of parents and
        children.

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
    # using the current node index, unwrap parameters and parents
    value_parent_idxs = edges[node_idx].value_parents

    # Return here if no parents node are found
    if value_parent_idxs is None:
        return attributes

    #############################################
    # Update the continuous value parents (x-2) #
    #############################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                (
                    pi_value_parent,
                    mu_value_parent,
                ) = prediction_error_value_parent(attributes, edges, value_parent_idx)
                # 5. Update node's parameters and node's parents recursively
                attributes[value_parent_idx]["precision"] = pi_value_parent
                attributes[value_parent_idx]["mean"] = mu_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_prediction(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the expected mean and precision of a binary state node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility of parents and
        children.

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
    attributes[node_idx]["expected_precision"] = expected_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction_error(
    attributes: Dict,
    time_step: float,
    value: float,
    node_idx: int,
    edges: Edges,
) -> Dict:
    """Update the input node structure given one binary observation.

    This function is the entry level of the binary node. It updates the parents of
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
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility of parents and
        children.
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
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step

    # list the value parents of the binary input
    value_parent_idxs = edges[node_idx].value_parents

    if value_parent_idxs is None:
        return attributes

    ####################################################
    # Update the value parent of the binary input node #
    ####################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            (
                precision_value_parent,
                mean_value_parent,
                surprise,
            ) = prediction_error_input_value_parent(attributes, edges, value_parent_idx)

            # Update value parent's parameters
            attributes[value_parent_idx]["precision"] = precision_value_parent
            attributes[value_parent_idx]["mean"] = mean_value_parent

    attributes[node_idx]["surprise"] = surprise

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction_error_missing_input(
    attributes: Dict,
    time_step: float,
    value: float,
    node_idx: int,
    edges: Edges,
) -> Dict:
    """Update the binary parents when the observation is missing in the binary input.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    value :
        The new observed value.
    node_idx :
        Pointer to the binary input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index list the value and volatility of parents and
        children.

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
    blank_fn = Partial(binary_input_blank_update, node_idx=node_idx, edges=edges)

    regular_fn = Partial(binary_input_prediction_error, node_idx=node_idx, edges=edges)

    # if the observation is missing, use the blank update
    # otherwise use the regular update for binary inputs
    attributes = cond(
        jnp.isnan(value), blank_fn, regular_fn, attributes, time_step, value
    )

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_blank_update(
    attributes: Dict,
    time_step: float,
    value: float,
    node_idx: int,
    edges: Edges,
):
    """Update of a binary input node in the case of missing observation."""
    # when the binary input is missing, this step does not update the parent node
    # only the continuous value parent of the binary state node will be updated
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step
    attributes[node_idx]["surprise"] = 0.0

    return attributes
