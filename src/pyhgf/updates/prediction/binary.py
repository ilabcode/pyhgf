# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

from jax import Array, jit

from pyhgf.math import sigmoid
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_state_node_prediction(
    attributes: Dict, edges: Edges, node_idx: int, **args
) -> Tuple[Array, ...]:
    r"""Get the new expected mean and precision of a binary state node.

    The predictions of a binary state node :math:`b` at time :math:`k` depends on the
    prediction of its value parent :math:`a`, such as:

    .. math::

        \hat{\mu}_b^{(k)} = \frac{1}{1 + e^{-\hat{\mu}_a^{(k)}}}

    and

    .. math::

        \hat{\pi}_b^{(k)} = \frac{1}{\hat{\mu}^{(k)}(1-\hat{\mu}^{(k)})}

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
    node_idx :
        Pointer to the binary state node.

    Returns
    -------
    expected_precision :
        The new expected precision.
    expected_mean :
        The mean expected mean.

    """
    # List the (unique) value parent of the binary state node
    value_parent_idx = edges[node_idx].value_parents[0]  # type: ignore

    # Estimate the new expected mean using the sigmoid transform
    expected_mean = attributes[value_parent_idx]["expected_mean"]
    expected_mean = sigmoid(expected_mean)

    # Estimate the new expected precision from the new expected mean
    expected_precision = 1 / (expected_mean * (1 - expected_mean))

    # use the inverse to fit the posterio update of the value parent
    # (eq. 97, Weber et al., v1)
    attributes[node_idx]["expected_precision"] = 1 / expected_precision

    # Update this node's parameters - use different parameter names as these values are
    # not used for the update of the value parent of the binary state node
    attributes[node_idx]["binary_expected_precision"] = expected_precision
    attributes[node_idx]["binary_expected_mean"] = expected_mean

    return expected_precision, expected_mean
