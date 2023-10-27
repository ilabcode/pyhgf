# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.prediction.continuous import predict_mean, predict_precision
from pyhgf.updates.prediction_error.continuous import (
    prediction_error_input_mean_value_parent,
    prediction_error_mean_value_parent,
    prediction_error_mean_volatility_parent,
    prediction_error_precision_value_parent,
    prediction_error_precision_volatility_parent,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction_error(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Prediction-error step for the value and volatility parents of a continuous node.

    Updating the node's parents is a two-step process:
    1. Update value parent(s).
    2. Update volatility parent(s).

    If a value/volatility parent has multiple children, all the children will update
    the parent together, therefor this function should only be called once per group
    of child nodes. The method :py:meth:`pyhgf.model.HGF.get_update_sequence`
    ensures that this function is only called once all the children have been
    updated.

    Then returns the structure of the new parameters.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
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
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # list value and volatility parents
    value_parents_idxs = edges[node_idx].value_parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # return here if no parents node are provided
    if (value_parents_idxs is None) and (volatility_parents_idxs is None):
        return attributes

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                # Estimate the precision of the posterior distribution
                precision_value_parent = prediction_error_precision_value_parent(
                    attributes, edges, value_parent_idx
                )
                # Estimate the mean of the posterior distribution
                mean_value_parent = prediction_error_mean_value_parent(
                    attributes, edges, value_parent_idx, precision_value_parent
                )

                # Update this parent's parameters
                attributes[value_parent_idx]["precision"] = precision_value_parent
                attributes[value_parent_idx]["mean"] = mean_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        for volatility_parent_idx in volatility_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[volatility_parent_idx].volatility_children[-1] == node_idx:
                # Estimate the new precision of the volatility parent
                precision_volatility_parent = (
                    prediction_error_precision_volatility_parent(
                        attributes, edges, time_step, volatility_parent_idx
                    )
                )
                # Estimate the new mean of the volatility parent
                mean_volatility_parent = prediction_error_mean_volatility_parent(
                    attributes,
                    edges,
                    time_step,
                    volatility_parent_idx,
                    precision_volatility_parent,
                )

                # Update this parent's parameters
                attributes[volatility_parent_idx][
                    "precision"
                ] = precision_volatility_parent
                attributes[volatility_parent_idx]["mean"] = mean_volatility_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def ehgf_continuous_node_prediction_error(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Perform the eHGF PE step for value and volatility parents of a continuous node.

    This update step uses a different order for the mean and precision as compared to
    the standard HGF, respectively:
    1. Update volatility parent(s).
    2. Update value parent(s).

    If a value/volatility parent has multiple children, all the children will update
    the parent together, therefor this function should only be called once per group
    of child nodes. The method :py:meth:`pyhgf.model.HGF.get_update_sequence`
    ensures that this function is only called once all the children have been
    updated.

    Then returns the structure of the new parameters.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
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
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # list value and volatility parents
    value_parents_idxs = edges[node_idx].value_parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # return here if no parents node are provided
    if (value_parents_idxs is None) and (volatility_parents_idxs is None):
        return attributes

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                # Estimate the precision of the posterior distribution
                precision_value_parent = prediction_error_precision_value_parent(
                    attributes, edges, value_parent_idx
                )
                # Estimate the mean of the posterior distribution
                mean_value_parent = prediction_error_mean_value_parent(
                    attributes, edges, value_parent_idx, precision_value_parent
                )

                # Update this parent's parameters
                attributes[value_parent_idx]["precision"] = precision_value_parent
                attributes[value_parent_idx]["mean"] = mean_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        for volatility_parent_idx in volatility_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[volatility_parent_idx].volatility_children[-1] == node_idx:
                # in the eHGF update step, we use the expected precision here
                # as we haven't computed it yet due to the reverse update order
                precision_volatility_parent = attributes[volatility_parent_idx][
                    "expected_precision"
                ]

                # Estimate the new mean of the volatility parent
                mean_volatility_parent = prediction_error_mean_volatility_parent(
                    attributes,
                    edges,
                    time_step,
                    volatility_parent_idx,
                    precision_volatility_parent,
                )
                attributes[volatility_parent_idx]["mean"] = mean_volatility_parent

                # Estimate the new precision of the volatility parent
                precision_volatility_parent = (
                    prediction_error_precision_volatility_parent(
                        attributes, edges, time_step, volatility_parent_idx
                    )
                )

                # Update this parent's parameters
                attributes[volatility_parent_idx][
                    "precision"
                ] = precision_volatility_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the expected mean and precision of a continuous node.

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
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that will be updated.
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
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the new expected mean
    expected_mean = predict_mean(attributes, edges, time_step, node_idx)
    # Get the new expected precision
    expected_precision = predict_precision(attributes, edges, time_step, node_idx)

    # Update this node's parameters
    attributes[node_idx]["expected_precision"] = expected_precision
    attributes[node_idx]["expected_mean"] = expected_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_prediction_error(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Update the input node structure.

    This function is the entry-level of the structure updates. It updates the parent
    of the input node.

    Parameters
    ----------
    value :
        The new observed value.
    time_step :
        The interval between the previous time point and the current time point.
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
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
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_update, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step

    # list value and volatility parents
    value_parents_idxs = edges[node_idx].value_parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                # Estimate the new precision of the value parent
                pi_value_parent = prediction_error_precision_value_parent(
                    attributes, edges, value_parent_idx
                )
                # Estimate the new mean of the value parent
                mu_value_parent = prediction_error_input_mean_value_parent(
                    attributes, edges, value_parent_idx, pi_value_parent
                )

                # update input node's parameters
                attributes[value_parent_idx]["precision"] = pi_value_parent
                attributes[value_parent_idx]["mean"] = mu_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        for volatility_parent_idx in volatility_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[volatility_parent_idx].volatility_children[-1] == node_idx:
                # in the eHGF update step, we use the expected precision here
                # as we haven't computed it yet due to the reverse update order
                precision_volatility_parent = attributes[volatility_parent_idx][
                    "expected_precision"
                ]

                # Estimate the new mean of the volatility parent
                mean_volatility_parent = prediction_error_mean_volatility_parent(
                    attributes,
                    edges,
                    time_step,
                    volatility_parent_idx,
                    precision_volatility_parent,
                )
                attributes[volatility_parent_idx]["mean"] = mean_volatility_parent

                # Estimate the new precision of the volatility parent
                precision_volatility_parent = (
                    prediction_error_precision_volatility_parent(
                        attributes, edges, time_step, volatility_parent_idx
                    )
                )

                # Update this parent's parameters
                attributes[volatility_parent_idx][
                    "precision"
                ] = precision_volatility_parent

    return attributes
