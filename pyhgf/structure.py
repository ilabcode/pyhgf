# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.typing import ArrayLike

from pyhgf.typing import NodeStructure, UpdateSequence


@partial(jit, static_argnames=("update_sequence", "node_structure", "input_nodes_idx"))
def beliefs_propagation(
    parameters_structure: Dict,
    data: ArrayLike,
    update_sequence: UpdateSequence,
    node_structure: Tuple,
    input_nodes_idx: Tuple = (0,),
) -> Tuple[Dict, Dict]:
    """Update the network's parameters after observing new data point(s).

    This function performs the beliefs propagation step at a time *t* triggered by the
    observation of a new batch of value(s). The way beliefs propagate is defined by the
    `update_sequence` and the `node_structure`. A tuple of two new
    `parameter_structure` is then returned (the carryover and the accumulated in the
    context of :py:func:`jax.lax.scan`).

    Parameters
    ----------
    parameters_structure :
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the beliefs propagation step.
    data :
        An array containing the new observation(s) as well as the time steps. The new
        observations can be a single value or a vector of observation with a length
        matching the length `input_nodes_idx`. `input_nodes_idx` is used to index the
        ith value in the vector to the ith input node, so the ordering of the input
        array matters. The time steps are the last column of the array, the default is
        unit incrementation.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    node_structure :
        The nodes structure.
    input_nodes_idx :
        An array of input indexes. The default is `[0]` (one input node, the first node
        in the network).

    """
    # extract value(s) and time steps from the data
    values = data[:-1]
    time_step = data[-1]
    input_nodes_idx = jnp.asarray(input_nodes_idx)

    # Fit the model with the current time and value variables, given model structure
    for sequence in update_sequence:
        node_idx, update_fn = sequence

        # if we are updating an input node, select the value that should be passed
        value = jnp.where(
            jnp.isin(node_idx, input_nodes_idx),
            jnp.sum(jnp.equal(input_nodes_idx, node_idx) * values),
            jnp.nan,
        )

        parameters_structure = update_fn(
            parameters_structure=parameters_structure,
            time_step=time_step,
            node_idx=node_idx,
            node_structure=node_structure,
            value=value,
        )

    return (
        parameters_structure,
        parameters_structure,
    )  # ("carryover", "accumulated")


def get_children(node_idx: int, node_structure: NodeStructure) -> List[int]:
    """Get the list of all children from a given node.

    Parameters
    ----------
    node_idx :
        Index of the parent node.
    node_structure :
        The network structure.

    Returns
    -------
    children_idx :
        The indexes of the children nodes.

    """
    children_idxs = []
    for idx in range(len(node_structure)):
        if np.any([node_idx in i for i in node_structure[idx] if i is not None]):
            children_idxs.append(idx)

    return children_idxs
