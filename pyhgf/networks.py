# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.typing import ArrayLike

from pyhgf.typing import UpdateSequence


@partial(jit, static_argnames=("update_sequence", "edges", "input_nodes_idx"))
def beliefs_propagation(
    attributes: Dict,
    data: ArrayLike,
    update_sequence: UpdateSequence,
    edges: Tuple,
    input_nodes_idx: Tuple = (0,),
) -> Tuple[Dict, Dict]:
    """Update the network's parameters after observing new data point(s).

    This function performs the beliefs propagation step at a time *t* triggered by the
    observation of a new batch of value(s). The way beliefs propagate is defined by the
    `update_sequence` and the `edges`. A tuple of two new
    `parameter_structure` is then returned (the carryover and the accumulated in the
    context of :py:func:`jax.lax.scan`).

    Parameters
    ----------
    attributes :
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
    edges :
        The nodes structure.
    input_nodes_idx :
        An array of input indexes. The default is `[0]` (one input node, the first node
        in the network).

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).


    """
    # extract value(s) and time steps from the data
    values = data[:-1]
    time_step = data[-1]

    input_nodes_idx = jnp.asarray(input_nodes_idx)
    # Fit the model with the current time and value variables, given the model structure
    for sequence in update_sequence:
        node_idx, update_fn = sequence

        # if we are updating an input node, select the value that should be passed
        value = jnp.where(
            jnp.isin(node_idx, input_nodes_idx),
            jnp.nansum(jnp.equal(input_nodes_idx, node_idx) * values),
            jnp.nan,
        )

        attributes = update_fn(
            attributes=attributes,
            time_step=time_step,
            node_idx=node_idx,
            edges=edges,
            value=value,
        )

    return (
        attributes,
        attributes,
    )  # ("carryover", "accumulated")


def trim_sequence(
    exclude_node_idxs: List, update_sequence: UpdateSequence, edges: Tuple
) -> UpdateSequence:
    """Remove steps from an update sequence that depends on a set of nodes.

    Parameters
    ----------
    exclude_node_idxs :
        A list of node indexes. The nodes can be input nodes or any other node in the
        network.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    edges :
        The nodes structure.

    Returns
    -------
    trimmed_update_sequence :
        The update sequence without the update steps for nodes depending on the root
        list.

    """
    # list the nodes that depend on the root indexes
    branch_list = list_branches(node_idxs=exclude_node_idxs, edges=edges)

    # remove the update steps that are targetting the excluded nodes
    trimmed_update_sequence = tuple(
        [seq for seq in update_sequence if seq[0] not in branch_list]
    )

    return trimmed_update_sequence


def list_branches(node_idxs: List, edges: Tuple, branch_list: List = []) -> List:
    """Return the branch of a network from a given set of root nodes.

    This function searches recursively and lists the parents above a given node. If all
    the children of a given parent are on the exclusion list, this parent is also
    excluded.

    Parameters
    ----------
    node_idxs :
        A list of node indexes. The nodes can be input nodes or any other node in the
        network.
    edges :
        The nodes structure.
    branch_list :
        The list of nodes that are already excluded (i.e )

    Returns
    -------
    branch_list :
        The list of node indexes that belong to the branch.

    """
    for idx in node_idxs:
        # add this node to the exclusion list
        branch_list.append(idx)
        all_parents = np.array(
            [
                i
                for i in [
                    edges[idx].value_parents,
                    edges[idx].volatility_parents,
                ]
                if i is not None
            ]
        ).flatten()
        for parent_idx in all_parents:
            # list the children for this node
            all_children = np.array(
                [
                    i
                    for i in [
                        edges[parent_idx].value_children,
                        edges[parent_idx].volatility_children,
                    ]
                    if i is not None
                ]
            ).flatten()
            # if this parent has only excluded children, add it to the exclusion list
            if np.all([i in branch_list for i in all_children]):
                branch_list = list_branches(
                    [parent_idx], edges, branch_list=branch_list
                )

    return branch_list
