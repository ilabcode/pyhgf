# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("node_idx", "edges"))
def generic_input_prediction_error(
    attributes: Dict,
    time_step: float,
    edges: Edges,
    node_idx: int,
    value: float,
    observed: bool,
    **args
) -> Dict:
    """Prediction error from a generic input node.

    A generic input node receive an observation and pass it as it is to its value
    parents.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
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

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["values"] = value
    attributes[node_idx]["observed"] = observed
    attributes[node_idx]["time_step"] = time_step

    value_parent_idxs = edges[node_idx].value_parents  # type: ignore

    for value_parent_idx in value_parent_idxs:  # type: ignore
        attributes[value_parent_idx]["values"] = value

    return attributes
