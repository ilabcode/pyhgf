# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
    precision_value_parent: ArrayLike,
) -> Array:
    r"""Send prediction-error and update the mean of a value parent (continuous).

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
        The updated value for the mean of the value parent (:math:`\\mu`).

    See Also
    --------
    prediction_error_precision_value_parent, prediction_error_input_mean_value_parent

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_value_parent = attributes[value_parent_idx]["expected_mean"]

    # Gather prediction errors from all child nodes if the parent has many children
    # This part corresponds to the sum of children for the multi-children situations
    children_prediction_errors = 0.0
    for child_idx, value_coupling in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["value_coupling_children"],
    ):
        child_value_prediction_error = (
            attributes[child_idx]["mean"] - attributes[child_idx]["expected_mean"]
        )
        expected_precision_child = attributes[child_idx]["expected_precision"]
        children_prediction_errors += (
            value_coupling * expected_precision_child * child_value_prediction_error
        ) / precision_value_parent

    # Estimate the new mean of the value parent
    mean_value_parent = expected_mean_value_parent + children_prediction_errors

    return mean_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def prediction_error_precision_value_parent(
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
    pi_value_parent :
        The updated value for the precision of the value parent (:math:`\\pi`).

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
    children_precisions = 0.0
    for child_idx, value_coupling in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["value_coupling_children"],
    ):
        expected_precision_child = attributes[child_idx]["expected_precision"]
        children_precisions += value_coupling**2 * expected_precision_child

    # Estimate new value for the precision of the value parent
    precision_value_parent = expected_precision_value_parent + children_precisions

    return precision_value_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_error_precision_volatility_parent(
    attributes: Dict, edges: Edges, time_step: float, volatility_parent_idx: int
) -> Array:
    r"""Send prediction-error and update the precision of the volatility parent.

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
    value_parent_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    precision_volatility_parent :
        The updated value for the mean of the value parent (:math:`\\pi`).

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
    children_volatility_precision = 0.0
    for child_idx, kappas_children in zip(
        edges[volatility_parent_idx].volatility_children,
        attributes[volatility_parent_idx]["volatility_coupling_children"],
    ):
        # Look at the (optional) volatility parents and update logvol accordingly
        logvol = attributes[child_idx]["tonic_volatility"]
        if edges[child_idx].volatility_parents is not None:
            for children_volatility_parents, volatility_coupling in zip(
                edges[child_idx].volatility_parents,
                attributes[child_idx]["volatility_coupling_parents"],
            ):
                logvol += (
                    volatility_coupling
                    * attributes[children_volatility_parents]["mean"]
                )

        # Compute new value for nu
        nu_children = time_step * jnp.exp(logvol)
        nu_children = jnp.where(nu_children > 1e-128, nu_children, jnp.nan)

        expected_precision_children = attributes[child_idx]["expected_precision"]
        child_precision = attributes[child_idx]["precision"]
        vope_children = (
            1 / attributes[child_idx]["precision"]
            + (attributes[child_idx]["mean"] - attributes[child_idx]["expected_mean"])
            ** 2
        ) * attributes[child_idx]["expected_precision"] - 1

        children_volatility_precision += (
            0.5
            * (kappas_children * nu_children * expected_precision_children) ** 2
            * (1 + (1 - 1 / (nu_children * child_precision)) * vope_children)
        )

    # Estimate the new precision of the volatility parent
    precision_volatility_parent = (
        expected_precision_volatility_parent + children_volatility_precision
    )
    precision_volatility_parent = jnp.where(
        precision_volatility_parent <= 0, jnp.nan, precision_volatility_parent
    )

    return precision_volatility_parent


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def prediction_error_mean_volatility_parent(
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
    for child_idx, kappas_children in zip(
        edges[volatility_parent_idx].volatility_children,
        attributes[volatility_parent_idx]["volatility_coupling_children"],
    ):
        # Look at the (optional) volatility parents and update logvol accordingly
        logvol = attributes[child_idx]["tonic_volatility"]
        if edges[child_idx].volatility_parents is not None:
            for children_volatility_parents, volatility_coupling in zip(
                edges[child_idx].volatility_parents,
                attributes[child_idx]["volatility_coupling_parents"],
            ):
                logvol += (
                    volatility_coupling
                    * attributes[children_volatility_parents]["mean"]
                )

        # Compute new value for nu
        nu_children = time_step * jnp.exp(logvol)
        nu_children = jnp.where(nu_children > 1e-128, nu_children, jnp.nan)

        expected_precision_children = attributes[child_idx]["expected_precision"]
        vope_children = (
            1 / attributes[child_idx]["precision"]
            + (attributes[child_idx]["mean"] - attributes[child_idx]["expected_mean"])
            ** 2
        ) * attributes[child_idx]["expected_precision"] - 1
        children_volatility_prediction_error += (
            0.5
            * kappas_children
            * nu_children
            * expected_precision_children
            / precision_volatility_parent
            * vope_children
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
    r"""Send value prediction-error to the mean of a continuous input's value parent.

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
    mean_value_parent :
        The updated value for the mean of the value parent (:math:`\\mu`).

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
    # Get the current expected mean for the volatility parent.
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes.
    expected_mean_value_parent = attributes[value_parent_idx]["expected_mean"]

    # gather PE updates from other input nodes
    # in the case of a multivariate descendency
    children_prediction_errors = 0.0
    for child_idx, value_coupling in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["value_coupling_children"],
    ):
        child_value_prediction_error = (
            attributes[child_idx]["value"] - expected_mean_value_parent
        )
        expected_precision_child = attributes[child_idx]["expected_precision"]
        children_prediction_errors += (
            value_coupling * expected_precision_child * child_value_prediction_error
        ) / precision_value_parent

    # Compute the new mean of the value parent
    mean_value_parent = expected_mean_value_parent + children_prediction_errors

    return mean_value_parent
