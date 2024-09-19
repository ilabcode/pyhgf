# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

from jax import jit


@partial(jit, static_argnames=("node_idx"))
def set_observation(
    attributes: Dict,
    node_idx: int,
    value: float,
    observed: int,
) -> Dict:
    r"""Add observations to the target node by setting the posterior to a given value.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic network.
    node_idx :
        Pointer to the input node.
    value :
        The new observed value.
    observed :
        Whether value was observed or not.

    Returns
    -------
    attributes :
        The attributes of the probabilistic network.

    """
    attributes[node_idx]["mean"] = value
    attributes[node_idx]["observed"] = observed

    return attributes
