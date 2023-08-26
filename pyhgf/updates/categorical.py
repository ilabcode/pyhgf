# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges
from pyhgf.updates.binary import binary_surprise


@partial(jit, static_argnames=("edges", "node_idx"))
def categorical_input_update(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Update the categorical input node given an array of binary observations.

    This function should be called **after** the update of the implied binary HGFs. It
    receive a `None` as the boolean observations are passed to the binary inputs
    directly. This update uses the expected probability of the binary input to update
    an implied Dirichlet distribution.

    Parameters
    ----------
    value :
        The new observed value. This here is ignored as the observed values are passed
        to the binary inputs directly.
    time_step :
        The interval between the previous time point and the current time point.
    attributes :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        `"psis"` is the value coupling strength. It should have the same length as the
        volatility parents' indexes. `"kappas"` is the volatility coupling strength.
        It should have the same length as the volatility parents' indexes.
    edges :
        Tuple of :py:class:`pyhgf.typing.Indexes` with the same length as node number.
        For each node, the index list value and volatility parents.
    node_idx :
        Pointer to the node that needs to be updated.

    Returns
    -------
    attributes :
        The updated parameters structure.

    See Also
    --------
    binary_input_update, continuous_input_update

    """
    # get the new expected values from the implied binary inputs (parents predictions)
    new_xi = jnp.array(
        [
            attributes[edges[vapa].value_parents[0]]["muhat"]
            for vapa in edges[node_idx].value_parents  # type: ignore
        ]
    )

    # the values observed at time k
    attributes[node_idx]["value"] = jnp.array(
        [
            attributes[idx]["value"]
            for idx in edges[node_idx].value_parents  # type: ignore
        ]
    )

    # the prediction error
    pe = attributes[node_idx]["value"] - attributes[node_idx]["xi"]

    # the differential of expectations (parents predictions at time k and k-1)
    delta_xi = new_xi - attributes[node_idx]["xi"]

    # compute nu, hyperparameter of the Dirichlet distribution (also learning rate)
    nu = (pe / delta_xi) - 1

    # prediction mean
    attributes[node_idx]["mu"] = ((nu * new_xi) + 1) / jnp.sum(((nu * new_xi) + 1))

    # save the current sufficient statistics
    attributes[node_idx]["xi"] = new_xi
    attributes[node_idx]["time_step"] = time_step
    attributes[node_idx]["surprise"] = binary_surprise(
        x=attributes[node_idx]["value"], muhat=attributes[node_idx]["xi"]
    )

    return attributes
