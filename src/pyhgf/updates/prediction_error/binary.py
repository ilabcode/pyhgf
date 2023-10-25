# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array
from jax.lax import cond
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise, gaussian_density
from pyhgf.typing import Edges


def prediction_error_mean_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
    pi_value_parent: ArrayLike,
) -> Array:
    r"""Send prediction-error and update the mean of a value parent (binary).

    .. note::
       This function has similarities with its continuous counterpart
       (:py:func:`pyhgf.update.prediction_error.continuous.prediction_error_mean_value_parent`),
       the only difference being that the prediction error are not weighted by the
       precision of the child nodes the same way.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value_parent_idx :
        Pointer to the node that will be updated.
    pi_value_parent :
        The precision of the value parent that has already been updated.

    Returns
    -------
    mu_value_parent :
        The expected value for the mean of the value parent (:math:`\\mu`).

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_value_parent = attributes["expected_mean"][value_parent_idx]

    # Gather prediction errors from all child nodes if the parent has many children
    # This part corresponds to the sum of children for the multi-children situations
    value_prediction_error_children = jnp.where(
        edges["value_children"][value_parent_idx], attributes["mean"], 0.0
    ) - jnp.where(
        edges["value_children"][value_parent_idx], attributes["expected_mean"], 0.0
    )
    value_coupling_children = attributes["value_coupling_children"][value_parent_idx]
    children_prediction_error = (
        (value_coupling_children * value_prediction_error_children) / pi_value_parent
    ).sum()

    # Estimate the new mean of the value parent
    mu_value_parent = expected_mean_value_parent + children_prediction_error

    return mu_value_parent


def prediction_error_precision_value_parent(
    attributes: Dict, edges: Edges, value_parent_idx: int
) -> Array:
    r"""Send prediction-error and update the precision of a value parent (binary).

    .. note::
       This function has similarities with its continuous counterpart
       (:py:func:`pyhgf.update.prediction_error.continuous.prediction_error_precision_value_parent`),
       the only difference being that the prediction error are not weighted by the
       precision of the child nodes the same way.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value_parent_idx :
        Pointer to the node that will be updated.

    Returns
    -------
    pi_value_parent :
        The expected value for the mean of the value parent (:math:`\\pi`).

    """
    # Get the current expected precision for the volatility parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_precision_value_parent = attributes["expected_precision"][value_parent_idx]

    # Gather precision updates from all children nodes if the parent has many children.
    # This part corresponds to the sum over children for the multi-children situations.
    expected_variance_children = jnp.where(
        edges["value_children"][value_parent_idx],
        1 / attributes["expected_precision"],
        0.0,
    )
    value_coupling_children = attributes["value_coupling_children"][value_parent_idx]
    children_precision = (value_coupling_children * expected_variance_children).sum()

    # Estimate new value for the precision of the value parent
    pi_value_parent = expected_precision_value_parent + children_precision

    return pi_value_parent


def prediction_error_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    """Update the mean and precision of the value parent of a binary node.

    Updating the posterior distribution of the value parent is a two-step process:
    #. Update the posterior precision using
    :py:func:`pyhgf.updates.prediction_error.binary.prediction_error_precision_value_parent`.
    #. Update the posterior mean value using
    :py:func:`pyhgf.updates.prediction_error.binary.prediction_error_mean_value_parent`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value_parent_idx :
        Pointer to the value parent node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    # Estimate the new precision of the value parent
    precision_value_parent = prediction_error_precision_value_parent(
        attributes, edges, value_parent_idx
    )
    # Estimate the new mean of the value parent
    mean_value_parent = prediction_error_mean_value_parent(
        attributes, edges, value_parent_idx, precision_value_parent
    )

    # Update the node's attributes
    attributes["precision"] = (
        attributes["precision"].at[value_parent_idx].set(precision_value_parent)
    )
    attributes["mean"] = attributes["mean"].at[value_parent_idx].set(mean_value_parent)

    return attributes


def prediction_error_input_value_parent(
    attributes: Dict,
    edges: Edges,
    value_parent_idx: int,
) -> Tuple[Array, ...]:
    r"""Update the mean and precision of the value parent of a binary input node.

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
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.
    surprise :
        The binary surprise from observing the new state.

    """
    # Get the current expected mean from the value parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    expected_mean_value_parent = attributes["expected_mean"][value_parent_idx]

    # Read parameters from the binary input (the only value child of the value parent)
    # Currently, only one binary input can be child of the binary node
    child_node_mask = edges["value_children"][value_parent_idx]

    eta0 = jnp.where(child_node_mask, attributes["eta0"], 0.0).sum()
    eta1 = jnp.where(child_node_mask, attributes["eta1"], 0.0).sum()
    expected_precision = jnp.where(
        child_node_mask, attributes["expected_precision"], 0.0
    ).sum()
    value = jnp.where(child_node_mask, attributes["value"], 0.0).sum()

    # Compute the surprise, new mean and precision
    mu_value_parent, pi_value_parent, surprise = cond(
        expected_precision == jnp.inf,
        input_surprise_inf,
        input_surprise_reg,
        (expected_precision, value, eta1, eta0, expected_mean_value_parent),
    )

    return pi_value_parent, mu_value_parent, surprise


def input_surprise_inf(op):
    """Apply special case if expected_precision is `jnp.inf`."""
    _, value, _, _, expected_mean_value_parent = op
    mu_value_parent = value
    pi_value_parent = jnp.inf
    surprise = binary_surprise(value, expected_mean_value_parent)

    return mu_value_parent, pi_value_parent, surprise


def input_surprise_reg(op):
    """Compute the surprise, mu_value_parent and pi_value_parent."""
    expected_precision, value, eta1, eta0, expected_mean_value_parent = op

    # Likelihood under eta1
    und1 = jnp.exp(-expected_precision / 2 * (value - eta1) ** 2)

    # Likelihood under eta0
    und0 = jnp.exp(-expected_precision / 2 * (value - eta0) ** 2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    mu_value_parent = (
        expected_mean_value_parent
        * und1
        / (expected_mean_value_parent * und1 + (1 - expected_mean_value_parent) * und0)
    )
    pi_value_parent = 1 / (mu_value_parent * (1 - mu_value_parent))

    # Surprise
    surprise = -jnp.log(
        expected_mean_value_parent * gaussian_density(value, eta1, expected_precision)
        + (1 - expected_mean_value_parent)
        * gaussian_density(value, eta0, expected_precision)
    )

    return mu_value_parent, pi_value_parent, surprise
