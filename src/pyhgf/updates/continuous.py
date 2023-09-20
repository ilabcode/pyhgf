# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.prediction.continuous import (
    prediction_input_value_parent,
    prediction_value_parent,
    prediction_volatility_parent,
)
from pyhgf.updates.prediction_error.continuous import (
    prediction_error_input_value_parent,
    prediction_error_value_parent,
    prediction_error_volatility_parent,
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
                (pi_value_parent, mu_value_parent) = prediction_error_value_parent(
                    attributes, edges, time_step, value_parent_idx
                )

                # Update this parent's parameters
                attributes[value_parent_idx]["pi"] = pi_value_parent
                attributes[value_parent_idx]["mu"] = mu_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        for volatility_parent_idx in volatility_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[volatility_parent_idx].volatility_children[-1] == node_idx:
                (
                    pi_volatility_parent,
                    mu_volatility_parent,
                ) = prediction_error_volatility_parent(
                    attributes, edges, time_step, volatility_parent_idx
                )

                # Update this parent's parameters
                attributes[volatility_parent_idx]["pi"] = pi_volatility_parent
                attributes[volatility_parent_idx]["mu"] = mu_volatility_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_prediction(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Prediction step for the value and volatility parents of a continuous node.

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
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
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
            (pihat_value_parent, muhat_value_parent) = prediction_value_parent(
                attributes, edges, time_step, value_parent_idx
            )

            # Update this parent's parameters
            attributes[value_parent_idx]["pihat"] = pihat_value_parent
            attributes[value_parent_idx]["muhat"] = muhat_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        for volatility_parent_idx in volatility_parents_idxs:
            (
                pihat_volatility_parent,
                muhat_volatility_parent,
            ) = prediction_volatility_parent(
                attributes, edges, time_step, volatility_parent_idx
            )

            # Update this parent's parameters
            attributes[volatility_parent_idx]["pihat"] = pihat_volatility_parent
            attributes[volatility_parent_idx]["muhat"] = muhat_volatility_parent

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
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
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

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                (
                    pi_value_parent,
                    mu_value_parent,
                ) = prediction_error_input_value_parent(
                    attributes, edges, time_step, value_parent_idx
                )

                # update input node's parameters
                attributes[value_parent_idx]["pi"] = pi_value_parent
                attributes[value_parent_idx]["mu"] = mu_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_prediction(
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
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
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
    # store timestep in the node's parameters
    attributes[node_idx]["time_step"] = time_step

    # list value and volatility parents
    value_parents_idxs = edges[node_idx].value_parents

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                pihat_value_parent, muhat_value_parent = prediction_input_value_parent(
                    attributes, edges, time_step, value_parent_idx
                )

                # update input node's parameters
                attributes[value_parent_idx]["pihat"] = pihat_value_parent
                attributes[value_parent_idx]["muhat"] = muhat_value_parent

    return attributes
