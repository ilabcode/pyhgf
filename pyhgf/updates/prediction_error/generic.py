# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx", "edges"))
def generic_state_prediction_error(
    attributes: Dict, edges: Edges, node_idx: int, **args
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

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    for value_parent_idx in edges[node_idx].value_parents:  # type: ignore
        attributes[value_parent_idx]["mean"] = attributes[node_idx]["mean"]

    return attributes
