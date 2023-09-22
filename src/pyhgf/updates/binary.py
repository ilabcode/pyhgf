# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.continuous import predict_mean, predict_precision
from pyhgf.updates.prediction.binary import predict_input_value_parent
from pyhgf.updates.prediction_error.binary import (
    prediction_error_input_value_parent,
    prediction_error_value_parent,
)


@partial(jit, static_argnames=("edges", "node_idx"))
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
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
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
                attributes[value_parent_idx]["pi"] = pi_value_parent
                attributes[value_parent_idx]["mu"] = mu_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_prediction(
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
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
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
    # using the current node index, unwrap parameters and parents
    value_parent_idxs = edges[node_idx].value_parents

    # Return here if no parents node are found
    if value_parent_idxs is None:
        return attributes

    ################################################################
    # Update the predictions of the continuous value parents (x-2) #
    ################################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            pihat_value_parent = predict_precision(
                attributes, edges, time_step, value_parent_idx
            )
            muhat_value_parent = predict_mean(
                attributes, edges, time_step, value_parent_idx
            )

            # update the parent nodes' parameters
            attributes[value_parent_idx]["pihat"] = pihat_value_parent
            attributes[value_parent_idx]["muhat"] = muhat_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
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
    .. note::
        `"psis"` is the value coupling strength. It should have the same length as the
        volatility parents' indexes. `"kappas"` is the volatility coupling strength.
        It should have the same length as the volatility parents' indexes.
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
    attributes[node_idx]["value"] = value
    attributes[node_idx]["time_step"] = time_step

    # list value and volatility parents
    value_parent_idxs = edges[node_idx].value_parents
    volatility_parent_idxs = edges[node_idx].volatility_parents

    if (value_parent_idxs is None) and (volatility_parent_idxs is None):
        return attributes

    ####################################################
    # Update the value parent of the binary input node #
    ####################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            (
                pi_value_parent,
                mu_value_parent,
                surprise,
            ) = prediction_error_input_value_parent(attributes, edges, value_parent_idx)

            # Update value parent's parameters
            attributes[value_parent_idx]["pi"] = pi_value_parent
            attributes[value_parent_idx]["mu"] = mu_value_parent

    attributes[node_idx]["surprise"] = surprise

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction(
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
    .. note::
        `"psis"` is the value coupling strength. It should have the same length as the
        volatility parents' indexes. `"kappas"` is the volatility coupling strength.
        It should have the same length as the volatility parents' indexes.
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
    # list value and volatility parents
    value_parent_idxs = edges[node_idx].value_parents
    volatility_parent_idxs = edges[node_idx].volatility_parents

    if (value_parent_idxs is None) and (volatility_parent_idxs is None):
        return attributes

    #######################################################
    # Update the value parent(s) of the binary input node #
    #######################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            pihat_value_parent, muhat_value_parent = predict_input_value_parent(
                attributes, edges, time_step, value_parent_idx
            )

            # Update value parent's parameters
            attributes[value_parent_idx]["pihat"] = pihat_value_parent
            attributes[value_parent_idx]["muhat"] = muhat_value_parent

    return attributes
