# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit

from pyhgf.typing import Edges

from .posterior_update_mean_continuous_node_unbounded import (
    posterior_update_mean_continuous_node_unbounded,
)
from .posterior_update_precision_continuous_node_unbounded import (
    posterior_update_precision_continuous_node_unbounded,
)


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_posterior_update_unbounded(
    attributes: Dict, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the posterior of a continuous node using an unbounded approximation.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_node_posterior_update_ehgf

    """
    # update the posterior mean and precision using the eHGF update step
    # we start with the mean update using the expected precision as an approximation
    posterior_precision, precision_l1, precision_l2 = (
        posterior_update_precision_continuous_node_unbounded(
            attributes,
            edges,
            node_idx,
        )
    )
    attributes[node_idx]["precision"] = posterior_precision

    posterior_mean = posterior_update_mean_continuous_node_unbounded(
        attributes=attributes,
        edges=edges,
        node_idx=node_idx,
        precision_l1=precision_l1,
        precision_l2=precision_l2,
    )
    attributes[node_idx]["mean"] = posterior_mean

    return attributes
