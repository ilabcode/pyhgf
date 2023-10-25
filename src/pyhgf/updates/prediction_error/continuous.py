# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("value_parent_idx"))
def prediction_error_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
    pi_value_parent: ArrayLike,
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
    pi_value_parent :
        The precision of the value parent that has already been updated.

    Returns
    -------
    mu_value_parent :
        The updated value for the mean of the value parent (:math:`\\mu`).

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_value_parent = attributes["expected_mean"][value_parent_idx]

    # Gather prediction errors from all child nodes if the parent has many children
    # This part corresponds to the sum of children for the multi-children situations
    pe_children = 0.0
    for child_idx, value_coupling in zip(
        jnp.where(edges["value_children"][value_parent_idx])[0],
        attributes["value_coupling_children"][value_parent_idx],
    ):
        vape_child = (
            attributes["mean"][child_idx] - attributes["expected_mean"][child_idx]
        )
        expected_precision_child = attributes["expected_precision"][child_idx]
        pe_children += (
            value_coupling * expected_precision_child * vape_child
        ) / pi_value_parent

    # Estimate the new mean of the value parent
    mu_value_parent = expected_mean_value_parent + pe_children

    return mu_value_parent


@partial(jit, static_argnames=("value_parent_idx"))
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

    """
    # Get the current expected mean for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_precision_value_parent = attributes["expected_precision"][value_parent_idx]

    # Gather precision updates from all child nodes if the parent has many children.
    # This part corresponds to the sum over children for the multi-children situations.
    pi_children = 0.0
    for child_idx, value_coupling in zip(
        jnp.where(edges["value_children"][value_parent_idx])[0],
        attributes["value_coupling_children"][value_parent_idx],
    ):
        expected_precision_child = attributes["expected_precision"][child_idx]
        pi_children += value_coupling**2 * expected_precision_child

    # Estimate new value for the precision of the value parent
    pi_value_parent = expected_precision_value_parent + pi_children

    return pi_value_parent


@partial(jit, static_argnames=("volatility_parent_idx"))
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
    pi_volatility_parent :
        The updated value for the mean of the value parent (:math:`\\pi`).

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_precision_volatility_parent = attributes["expected_precision"][
        volatility_parent_idx
    ]

    # gather volatility precisions from the child nodes
    children_volatility_precision = 0.0
    for child_idx, kappas_children in zip(
        jnp.where(edges["volatility_children"][volatility_parent_idx])[0],
        attributes["volatility_coupling_children"][volatility_parent_idx],
    ):
        # Look at the (optional) volatility parents and update logvol accordingly
        logvol = attributes["tonic_volatility"][child_idx]
        for child_vo_pa, volatility_coupling in zip(
            jnp.where(edges["volatility_parents"][child_idx])[0],
            attributes["volatility_coupling_parents"][child_idx],
        ):
            logvol += volatility_coupling * attributes["mean"][child_vo_pa]

        # Compute new value for nu
        nu_children = time_step * jnp.exp(logvol)
        nu_children = jnp.where(nu_children > 1e-128, nu_children, jnp.nan)

        expected_precision_children = attributes["expected_precision"][child_idx]
        pi_children = attributes["precision"][child_idx]
        vope_children = (
            1 / attributes["precision"][child_idx]
            + (attributes["mean"][child_idx] - attributes["expected_mean"][child_idx])
            ** 2
        ) * attributes["expected_precision"][child_idx] - 1

        children_volatility_precision += (
            0.5
            * (kappas_children * nu_children * expected_precision_children) ** 2
            * (1 + (1 - 1 / (nu_children * pi_children)) * vope_children)
        )

    # Estimate the new precision of the volatility parent
    pi_volatility_parent = (
        expected_precision_volatility_parent + children_volatility_precision
    )
    pi_volatility_parent = jnp.where(
        pi_volatility_parent <= 0, jnp.nan, pi_volatility_parent
    )

    return pi_volatility_parent


@partial(jit, static_argnames=("volatility_parent_idx"))
def prediction_error_mean_volatility_parent(
    attributes, edges, time_step, volatility_parent_idx, pi_volatility_parent: ArrayLike
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
    valueolatility_parent_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    mu_volatility_parent :
        The updated value for the mean of the value parent (:math:`\\mu`).

    """
    # Get the current expected mean for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_volatility_parent = attributes["expected_mean"][volatility_parent_idx]

    # Gather volatility prediction errors from the child nodes
    children_volatility_prediction_error = 0.0
    for child_idx, kappas_children in zip(
        jnp.where(edges["volatility_children"][volatility_parent_idx])[0],
        attributes["volatility_coupling_children"][volatility_parent_idx],
    ):
        # Look at the (optional) volatility parents of the child
        # and update logvol accordingly
        logvol = attributes["tonic_volatility"][child_idx]
        for child_volatility_paren_idx, volatility_coupling in zip(
            jnp.where(edges["volatility_parents"][child_idx])[0],
            attributes["volatility_coupling_parents"][child_idx],
        ):
            logvol += (
                volatility_coupling * attributes["mean"][child_volatility_paren_idx]
            )

        # Compute new value for nu
        nu_children = time_step * jnp.exp(logvol)
        nu_children = jnp.where(nu_children > 1e-128, nu_children, jnp.nan)

        expected_precision_children = attributes["expected_precision"][child_idx]
        vope_children = (
            1 / attributes["precision"][child_idx]
            + (attributes["mean"][child_idx] - attributes["expected_mean"][child_idx])
            ** 2
        ) * attributes["expected_precision"][child_idx] - 1
        children_volatility_prediction_error += (
            0.5
            * kappas_children
            * nu_children
            * expected_precision_children
            / pi_volatility_parent
            * vope_children
        )

    # Estimate the new mean of the volatility parent
    mu_volatility_parent = (
        expected_mean_volatility_parent + children_volatility_prediction_error
    )

    return mu_volatility_parent


@partial(jit, static_argnames=("value_parent_idx"))
def prediction_error_input_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
    pi_value_parent: ArrayLike,
) -> Array:
    r"""Send prediction-error to the mean of a continuous input's value parent.

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
    mu_value_parent :
        The updated value for the mean of the value parent (:math:`\\mu`).

    """
    # Get the current expected mean for the volatility parent.
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes.
    expected_mean_value_parent = attributes["expected_mean"][value_parent_idx]

    # gather PE updates from other input nodes
    # in the case of a multivariate descendency
    pe_children = 0.0
    for child_idx, value_coupling in zip(
        jnp.where(edges["value_children"][value_parent_idx])[0],
        attributes["value_coupling_children"][value_parent_idx],
    ):
        child_value_prediction_error = (
            attributes["value"][child_idx] - expected_mean_value_parent
        )
        expected_precision_child = attributes["expected_precision"][child_idx]
        pe_children += (
            value_coupling * expected_precision_child * child_value_prediction_error
        ) / pi_value_parent

    # Compute the new mean of the value parent
    mu_value_parent = expected_mean_value_parent + pe_children

    return mu_value_parent
