# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_input_precision_value_parent(
    attributes: Dict, edges: Edges, value_parent_idx: int
) -> Array:
    r"""Send prediction-error and update the precision of a value parent (continuous).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value_parent_idx :
        Pointer to the value parent node that will be updated.

    Returns
    -------
    precision_value_parent :
        The updated value for the precision of the value parent.

    See Also
    --------
    prediction_error_mean_value_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the current expected mean for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_precision_value_parent = attributes[value_parent_idx]["expected_precision"]

    # Gather precision updates from all child nodes if the parent has many children.
    # This part corresponds to the sum over children for the multi-children situations.
    expected_precision_children = 0.0
    for child_idx, value_coupling in zip(
        edges[value_parent_idx].value_children,  # type: ignore
        attributes[value_parent_idx]["value_coupling_children"],
    ):
        # the expected precision in the input node
        expected_precision_child = attributes[child_idx]["expected_precision"]

        # add the precision from volatility parents if any
        if edges[child_idx].volatility_parents is not None:
            volatility_coupling = attributes[edges[child_idx].volatility_parents[0]][
                "volatility_coupling_children"
            ][0]
            expected_precision_child *= 1 / jnp.exp(
                attributes[edges[child_idx].volatility_parents[0]]["expected_mean"]
                * volatility_coupling
            )

        expected_precision_children += expected_precision_child * value_coupling**2

    # Estimate new value for the precision of the value parent
    precision_value_parent = (
        expected_precision_value_parent + expected_precision_children
    )
    return precision_value_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_error_input_precision_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    """Send prediction-error and update the precision of the volatility parent.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The interval between the previous time point and the current time point.
    volatility_parent_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    precision_volatility_parent :
        The new precision of the value parent.

    See Also
    --------
    prediction_error_mean_volatility_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_precision_volatility_parent = attributes[volatility_parent_idx][
        "expected_precision"
    ]

    # gather volatility precisions from the child nodes
    precision_volatility_children = 0.0
    for child_idx, volatility_coupling in zip(
        edges[volatility_parent_idx].volatility_children,  # type: ignore
        attributes[volatility_parent_idx]["volatility_coupling_children"],
    ):
        # retireve the index of the value parent (assuming a unique value parent)
        # we need this to compute the value PE, required for the volatility PE
        this_value_parent_idx = edges[child_idx].value_parents[0]

        # compute the expected precision from the input node
        expected_precision_child = attributes[child_idx]["expected_precision"]

        # add the precision from the volatility parent if any
        expected_precision_child *= 1 / jnp.exp(
            attributes[edges[child_idx].volatility_parents[0]]["expected_mean"]
            * volatility_coupling
        )

        # compute the volatility prediction error for this input node
        child_volatility_prediction_error = (
            expected_precision_child / attributes[this_value_parent_idx]["precision"]
            + expected_precision_child
            * (
                attributes[child_idx]["value"]
                - attributes[this_value_parent_idx]["mean"]
            )
            ** 2
            - 1
        )

        precision_volatility_children += (
            0.5 * volatility_coupling**2 * (1 + child_volatility_prediction_error)
        )

    # Estimate the new precision of the volatility parent
    precision_volatility_parent = (
        expected_precision_volatility_parent + precision_volatility_children
    )
    precision_volatility_parent = jnp.where(
        precision_volatility_parent <= 0, jnp.nan, precision_volatility_parent
    )

    return precision_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_error_input_mean_volatility_parent(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    volatility_parent_idx: int,
    precision_volatility_parent: ArrayLike,
) -> Array:
    r"""Send prediction-error and update the mean of the volatility parent.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The interval between the previous time point and the current time point.
    volatility_parent_idx :
        Pointer to the node that will be updated.
    precision_volatility_parent :
        The precision of the volatility parent.

    Returns
    -------
    mean_volatility_parent :
        The updated value for the mean of the value parent (:math:`\\mu`).

    See Also
    --------
    prediction_error_volatility_volatility_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the current expected mean for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_volatility_parent = attributes[volatility_parent_idx]["expected_mean"]

    # Gather volatility prediction errors from the child nodes
    children_volatility_prediction_error = 0.0
    for child_idx, volatility_coupling in zip(
        edges[volatility_parent_idx].volatility_children,  # type: ignore
        attributes[volatility_parent_idx]["volatility_coupling_children"],
    ):
        # retireve the index of the value parent (assuming a unique value parent)
        # we need this to compute the value PE, required for the volatility PE
        this_value_parent_idx = edges[child_idx].value_parents[0]

        # compute the expected precision from the input node
        expected_precision_child = attributes[child_idx]["expected_precision"]

        # add the precision from the volatility parent if anyu
        expected_precision_child *= 1 / jnp.exp(
            attributes[edges[child_idx].volatility_parents[0]]["expected_mean"]
            * volatility_coupling
        )

        # compute the volatility prediction error for this input node
        child_volatility_prediction_error = (
            expected_precision_child / attributes[this_value_parent_idx]["precision"]
            + expected_precision_child
            * (
                attributes[child_idx]["value"]
                - attributes[this_value_parent_idx]["mean"]
            )
            ** 2
            - 1
        )

        # sum over all input nodes
        children_volatility_prediction_error += (
            0.5
            * child_volatility_prediction_error
            * (volatility_coupling / attributes[volatility_parent_idx]["precision"])
        )

    # Estimate the new mean of the volatility parent
    mean_volatility_parent = (
        expected_mean_volatility_parent + children_volatility_prediction_error
    )

    return mean_volatility_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_input_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
    precision_value_parent: ArrayLike,
) -> Array:
    """Send value prediction-error to the mean of a continuous input's value parent.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value_parent_idx :
        Pointer to the value parent node that will be updated.
    precision_value_parent :
        The precision of the value parent.

    Returns
    -------
    mean_value_parent :
        The new mean of the value parent.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    prediction_error_mean_value_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # gather PE updates from other input nodes
    # in the case of a multivariate descendency
    children_prediction_errors = 0.0
    for child_idx, value_coupling in zip(
        edges[value_parent_idx].value_children,  # type: ignore
        attributes[value_parent_idx]["value_coupling_children"],
    ):
        # retireve the index of the value parent (assuming a unique value parent)
        # we need this to compute the value PE, required for the volatility PE
        this_value_parent_idx = edges[child_idx].value_parents[0]

        child_value_prediction_error = (
            attributes[child_idx]["value"]
            - attributes[this_value_parent_idx]["expected_mean"]
        )

        expected_precision_child = attributes[child_idx]["expected_precision"]
        if edges[child_idx].volatility_parents is not None:
            volatility_coupling = attributes[edges[child_idx].volatility_parents[0]][
                "volatility_coupling_children"
            ][0]
            expected_precision_child *= 1 / jnp.exp(
                attributes[edges[child_idx].volatility_parents[0]]["expected_mean"]
                * volatility_coupling
            )

        children_prediction_errors += (
            (expected_precision_child / precision_value_parent)
            * child_value_prediction_error
            * value_coupling
        )

    # Compute the new mean of the value parent
    mean_value_parent = (
        attributes[value_parent_idx]["expected_mean"] + children_prediction_errors
    )

    return mean_value_parent
