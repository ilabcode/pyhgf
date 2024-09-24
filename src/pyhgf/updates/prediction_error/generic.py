# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx", "edges"))
def generic_state_prediction_error(
    attributes: Dict, edges: Edges, node_idx: int, value: float, observed: bool, **args
) -> Dict:
    """Prediction error from a generic input node.

    A generic input node receive an observation and pass it as it is to its value
    parents.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    value :
        The new observed value.
    observed :
        Whether value was observed or not.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["mean"] = value
    attributes[node_idx]["observed"] = observed

    for value_parent_idx in edges[node_idx].value_parents:  # type: ignore
        attributes[value_parent_idx]["mean"] = value

    return attributes
