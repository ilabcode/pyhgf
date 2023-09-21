# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.lax import cond
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise, gaussian_density
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
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
    muhat_value_parent = attributes[value_parent_idx]["muhat"]

    # Gather prediction errors from all child nodes if the parent has many children
    # This part corresponds to the sum of children for the multi-children situations
    pe_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        vape_child = attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]
        pe_children += (psi_child * vape_child) / pi_value_parent

    # Estimate the new mean of the value parent
    mu_value_parent = muhat_value_parent + pe_children

    return mu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
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
    pihat_value_parent = attributes[value_parent_idx]["pihat"]

    # Gather precision updates from all child nodes if the parent has many children.
    # This part corresponds to the sum over children for the multi-children situations.
    pi_children = 0.0
    for child_idx, psi_child in zip(
        edges[value_parent_idx].value_children,
        attributes[value_parent_idx]["psis_children"],
    ):
        pihat_child = attributes[child_idx]["pihat"]
        pi_children += psi_child * (1 / pihat_child)

    # Estimate new value for the precision of the value parent
    pi_value_parent = pihat_value_parent + pi_children

    return pi_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
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
    pi_value_parent :
        The precision (:math:`\\pi`) of the value parent.
    mu_value_parent :
        The mean (:math:`\\mu`) of the value parent.

    """
    # Estimate the new precision of the value parent
    pi_value_parent = prediction_error_precision_value_parent(
        attributes, edges, value_parent_idx
    )
    # Estimate the new mean of the value parent
    mu_value_parent = prediction_error_mean_value_parent(
        attributes, edges, value_parent_idx, pi_value_parent
    )

    return pi_value_parent, mu_value_parent


@partial(jit, static_argnames=("edges", "value_parent_idx"))
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
    # Get the current expected mean for the value parent
    # The prediction sequence was triggered by the new observation so this value is
    # already in the node attributes
    muhat_value_parent = attributes[value_parent_idx]["muhat"]

    # Read parameters from the binary input
    # Currently, only one binary input can be child of the binary node
    child_node_idx = edges[value_parent_idx].value_children[0]
    eta0 = attributes[child_node_idx]["eta0"]
    eta1 = attributes[child_node_idx]["eta1"]
    pihat = attributes[child_node_idx]["pihat"]
    value = attributes[child_node_idx]["value"]

    # Compute the surprise, new mean and precision
    mu_value_parent, pi_value_parent, surprise = cond(
        pihat == jnp.inf,
        input_surprise_inf,
        input_surprise_reg,
        (pihat, value, eta1, eta0, muhat_value_parent),
    )

    return pi_value_parent, mu_value_parent, surprise


def input_surprise_inf(op):
    """Apply special case if pihat is `jnp.inf` (just pass the value through)."""
    _, value, _, _, muhat_value_parent = op
    mu_value_parent = value
    pi_value_parent = jnp.inf
    surprise = binary_surprise(value, muhat_value_parent)

    return mu_value_parent, pi_value_parent, surprise


def input_surprise_reg(op):
    """Compute the surprise, mu_value_parent and pi_value_parent."""
    pihat, value, eta1, eta0, muhat_value_parent = op

    # Likelihood under eta1
    und1 = jnp.exp(-pihat / 2 * (value - eta1) ** 2)

    # Likelihood under eta0
    und0 = jnp.exp(-pihat / 2 * (value - eta0) ** 2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    mu_value_parent = (
        muhat_value_parent
        * und1
        / (muhat_value_parent * und1 + (1 - muhat_value_parent) * und0)
    )
    pi_value_parent = 1 / (mu_value_parent * (1 - mu_value_parent))

    # Surprise
    surprise = -jnp.log(
        muhat_value_parent * gaussian_density(value, eta1, pihat)
        + (1 - muhat_value_parent) * gaussian_density(value, eta0, pihat)
    )

    return mu_value_parent, pi_value_parent, surprise
