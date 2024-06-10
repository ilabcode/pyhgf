# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Callable, Dict

from jax import jit

from pyhgf.typing import Attributes, Edges


@partial(jit, static_argnames=("edges", "node_idx", "sufficient_stats_fn"))
def posterior_update_exponential_family(
    attributes: Dict, edges: Edges, node_idx: int, sufficient_stats_fn: Callable, **args
) -> Attributes:
    r"""Update the parameters of an exponential family distribution.

    Assuming that :math:`nu` is fixed, updating the hyperparameters of the distribution
    is given by:

    .. math::
        \xi \leftarrow \xi + \frac{1}{\nu + 1}(t(x)-\xi)

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
    sufficient_stats_fn :
        Compute the sufficient statistics of the probability distribution. This should
        be one of the method implemented in the distribution class in
        :py:class:`pyhgf.math.Normal`, for a univariate normal.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    References
    ----------
    .. [1] Mathys, C., & Weber, L. (2020). Hierarchical Gaussian Filtering of Sufficient
    Statistic Time Series for Active Inference. In Active Inference (pp. 52–58).
    Springer International Publishing. https://doi.org/10.1007/978-3-030-64919-7_7

    """
    # update the hyperparameter vectors
    attributes[node_idx]["xis"] = attributes[node_idx]["xis"] + (
        1 / (1 + attributes[node_idx]["nus"])
    ) * (
        sufficient_stats_fn(attributes[node_idx]["values"])
        - attributes[node_idx]["xis"]
    )

    return attributes
