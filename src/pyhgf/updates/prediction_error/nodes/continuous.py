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
        edges[value_parent_idx].value_children,  # type: ignore
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
        edges[value_parent_idx].value_children,  # type: ignore
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
    r"""Update the precision of the volatility parent.

    The new precision of the volatility parent :math:`a` of a state node at time
    :math:`k` is given by:

    .. math::

        \pi_a^{(k)} = \hat{\pi}_a^{(k)} + \sum_{j=1}^{N_{children}} \\
        \frac{1}{2} \left( \kappa_j \gamma_j^{(k)} \right) ^2 + \\
        \left( \kappa_j \gamma_j^{(k)} \right) ^2 \Delta_j^{(k)} - \\
        \frac{1}{2} \kappa_j^2 \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} + \\
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the volatility-weighted precision of the prediction,
    given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    with :math:`\Omega_j^{(k)}` the predicted volatility computed in the prediction
    step (:func:`pyhgf.updates.prediction.predict_precision`).

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
    # Get the current expected precision for the volatility parent - this assumes the
    # prediction sequence has already run and this value is in the node's attributes
    expected_precision_volatility_parent = attributes[volatility_parent_idx][
        "expected_precision"
    ]

    # gather the volatility precisions from all the child nodes
    children_volatility_precision = 0.0
    for child_idx, volatility_coupling in zip(
        edges[volatility_parent_idx].volatility_children,  # type: ignore
        attributes[volatility_parent_idx]["volatility_coupling_children"],
    ):
        # retrieve the predicted volatility (Ω) computed in the prediction step
        predicted_volatility = attributes[child_idx]["temp"]["predicted_volatility"]

        # compute the volatility weigthed precision (γ)
        volatility_weigthed_precision = (
            predicted_volatility * attributes[child_idx]["expected_precision"]
        )

        # compute the volatility prediction error (VOPE)
        vope_children = (
            (
                attributes[child_idx]["expected_precision"]
                / attributes[child_idx]["precision"]
            )
            + attributes[child_idx]["expected_precision"]
            * (attributes[child_idx]["mean"] - attributes[child_idx]["expected_mean"])
            ** 2
            - 1
        )

        # sum over all volatility children
        children_volatility_precision += (
            0.5 * (volatility_coupling * volatility_weigthed_precision) ** 2
            + (volatility_coupling * volatility_weigthed_precision) ** 2 * vope_children
            - 0.5
            * volatility_coupling**2
            * volatility_weigthed_precision
            * vope_children
        )

    # compute the new precision of the volatility parent
    precision_volatility_parent = (
        expected_precision_volatility_parent + children_volatility_precision
    )

    # ensure the new precision is greater than 0
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
    """Send prediction-error and update the mean of the volatility parent.

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
        The updated value for the mean of the value parent.

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

        vope_children = (
            1 / attributes[child_idx]["precision"]
            + (attributes[child_idx]["mean"] - attributes[child_idx]["expected_mean"])
            ** 2
        ) * attributes[child_idx]["expected_precision"] - 1
        children_volatility_prediction_error += (
            0.5
            * volatility_coupling
            * nu_children
            * attributes[child_idx]["expected_precision"]
            / precision_volatility_parent
            * vope_children
        )

    # Estimate the new mean of the volatility parent
    mean_volatility_parent = (
        expected_mean_volatility_parent + children_volatility_prediction_error
    )

    return mean_volatility_parent
