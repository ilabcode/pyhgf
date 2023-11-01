# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit
from jax.lax import cond

from pyhgf.math import binary_surprise, gaussian_density
from pyhgf.typing import Edges


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
    expected_mean_value_parent = attributes[value_parent_idx]["expected_mean"]

    # Read parameters from the binary input
    # Currently, only one binary input can be child of the binary node
    child_node_idx = edges[value_parent_idx].value_children[0]  # type: ignore
    eta0 = attributes[child_node_idx]["eta0"]
    eta1 = attributes[child_node_idx]["eta1"]
    expected_precision = attributes[child_node_idx]["expected_precision"]
    value = attributes[child_node_idx]["value"]

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
