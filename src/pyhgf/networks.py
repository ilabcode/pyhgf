# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import TYPE_CHECKING, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.typing import ArrayLike

from pyhgf.math import gaussian_surprise
from pyhgf.typing import Indexes, UpdateSequence
from pyhgf.updates.binary import (
    binary_input_prediction,
    binary_input_prediction_error,
    binary_node_prediction,
    binary_node_prediction_error,
)
from pyhgf.updates.categorical import categorical_input_update
from pyhgf.updates.continuous import (
    continuous_input_prediction,
    continuous_input_prediction_error,
    continuous_node_prediction,
    continuous_node_prediction_error,
)

if TYPE_CHECKING:
    from pyhgf.model import HGF


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
        # otherwise, just pass 0.0 and the value will be ignored
        value = jnp.sum(jnp.equal(input_nodes_idx, node_idx) * values)

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


def fill_categorical_state_node(
    hgf: "HGF", node_idx: int, implied_binary_parameters: Dict
) -> "HGF":
    """Generate a binary network implied by categorical state(-transition) nodes.

    Parameters
    ----------
    hgf :
        Instance of a HGF model.
    node_idx :
        Index to the categorical state node.
    implied_binary_parameters :
        Parameters for the set of implied binary HGFs.

    Returns
    -------
    hgf :
        The updated instance of the HGF model.

    """
    # add thew binary inputs
    hgf.add_input_node(
        kind="binary",
        input_idxs=implied_binary_parameters["binary_idxs"],
        binary_parameters={
            key: implied_binary_parameters[key]
            for key in ["eta0", "eta1", "binary_precision"]
        },
    )

    # add the value dependency between the categorical and binary nodes
    edges_as_list: List[Indexes] = list(hgf.edges)
    edges_as_list[node_idx] = Indexes(
        tuple(implied_binary_parameters["binary_idxs"]), None, None, None
    )
    for binary_idx in implied_binary_parameters["binary_idxs"]:
        edges_as_list[binary_idx] = Indexes(None, None, (node_idx,), None)
    hgf.edges = tuple(edges_as_list)

    # loop over the number of categories and create as many second-levels binary HGF
    for i in range(implied_binary_parameters["n_categories"]):
        # binary state node
        hgf.add_value_parent(
            children_idxs=implied_binary_parameters["binary_idxs"][i],
            value_coupling=1.0,
            pi=implied_binary_parameters["pi_1"],
            mu=implied_binary_parameters["mu_1"],
        )

    # add the continuous parent node
    for i in range(implied_binary_parameters["n_categories"]):
        hgf.add_value_parent(
            children_idxs=implied_binary_parameters["binary_idxs"][i]
            + implied_binary_parameters["n_categories"],
            value_coupling=1.0,
            mu=implied_binary_parameters["mu_2"],
            pi=implied_binary_parameters["pi_2"],
            omega=implied_binary_parameters["omega_2"],
        )

    # add the higher level volatility parents
    # as a shared parents between the second level nodes
    hgf.add_volatility_parent(
        children_idxs=[
            idx + 2 * implied_binary_parameters["n_categories"]
            for idx in implied_binary_parameters["binary_idxs"]
        ],
        volatility_coupling=1.0,
        mu=implied_binary_parameters["mu_3"],
        pi=implied_binary_parameters["pi_3"],
        omega=implied_binary_parameters["omega_3"],
    )

    return hgf


def get_update_sequence(
    hgf: "HGF",
    node_idxs: Tuple[int, ...] = (0,),
    update_sequence: List = [],
    prediction_sequence: List = [],
    init: bool = True,
) -> Tuple[List, ...]:
    """Generate an update sequence from the network's structure.

    THis function return an optimized update sequence considering the edges of the
    network. The function ensures that the following principles apply:
    1. all children have been updated before propagating to the parent(s) node.
    2. the update function of an input node is chosen based on the node's type
    (`"continuous"`, `"binary"` or `"categorical"`).
    3. the update function of the parent of an input node is chosen based on the
    node's type (`"continuous"`, `"binary"` or `"categorical"`).

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    node_idxs :
        The indexes of the current node(s) that should be evaluated. By default
        (`None`), the function will start with the list of input nodes.
    update_sequence :
        The current state of the update sequence (for recursive evaluation).
    prediction_sequence :
        The current state of the prediction sequence (for recursive evaluation).
    init :
        Use `init=False` for recursion.

    Returns
    -------
    sequence :
        The update sequence generated from the node structure.

    """
    if len(update_sequence) == 0:
        # list all input nodes except categoricals
        # that will be handled at the end of the recursion
        node_idxs = tuple(
            [
                idx
                for kind, idx in zip(hgf.input_nodes_idx.kind, hgf.input_nodes_idx.idx)
                if kind != "categorical"
            ]
        )

    for node_idx in node_idxs:
        # if the node has no parent, exit here
        if (hgf.edges[node_idx].value_parents is None) & (
            hgf.edges[node_idx].volatility_parents is None
        ):
            continue

        # if this node is part of a familly (the parent has multiple children)
        # and this is not the last of the children, exit here
        # we apply this principle for every value / volatility parent
        is_youngest = False
        if node_idx in hgf.input_nodes_idx.idx:
            is_youngest = True  # always update input nodes
        if hgf.edges[node_idx].value_parents is not None:
            for value_parents in hgf.edges[node_idx].value_parents:  # type: ignore
                if hgf.edges[value_parents].value_children[-1] == node_idx:
                    is_youngest = True
        if hgf.edges[node_idx].volatility_parents is not None:
            for volatility_parents in hgf.edges[
                node_idx
            ].volatility_parents:  # type: ignore
                if hgf.edges[volatility_parents].volatility_children[-1] == node_idx:
                    is_youngest = True

        # select the update function
        # --------------------------

        # case 1 - default to a continuous node
        update_fn = continuous_node_prediction_error
        prediction_fn = continuous_node_prediction

        # case 2 - this is an input node
        if node_idx in hgf.input_nodes_idx.idx:
            model_kind = [
                kind_
                for kind_, idx_ in zip(
                    hgf.input_nodes_idx.kind, hgf.input_nodes_idx.idx
                )
                if idx_ == node_idx
            ][0]
            if model_kind == "binary":
                update_fn = binary_input_prediction_error
                prediction_fn = binary_input_prediction
            elif model_kind == "continuous":
                update_fn = continuous_input_prediction_error
                prediction_fn = continuous_input_prediction
            elif model_kind == "categorical":
                continue

        # case 3 - this is the value parent of at least one binary input node
        if hgf.edges[node_idx].value_children is not None:
            value_children_idx = hgf.edges[node_idx].value_children
            for child_idx in value_children_idx:  # type: ignore
                if child_idx in hgf.input_nodes_idx.idx:
                    model_kind = [
                        kind_
                        for kind_, idx_ in zip(
                            hgf.input_nodes_idx.kind, hgf.input_nodes_idx.idx
                        )
                        if idx_ == child_idx
                    ][0]
                    if model_kind == "binary":
                        update_fn = binary_node_prediction_error
                        prediction_fn = binary_node_prediction

        # create a new update and prediction sequence step and add it to the list
        # only the youngest of the family is updated, but all nodes get predictions
        if is_youngest:
            new__update_sequence = node_idx, update_fn
            update_sequence.append(new__update_sequence)

        new_prediction_sequence = node_idx, prediction_fn
        prediction_sequence.append(new_prediction_sequence)

        # search recursively for the next update steps - make sure that all the
        # children have been updated before updating the parent(s)
        # ---------------------------------------------------------------------

        # loop over all possible dependencies (e.g. value, volatility coupling)
        parents = hgf.edges[node_idx][:2]
        for parent_types in parents:
            if parent_types is not None:
                for idx in parent_types:
                    # get the list of all the children in the parent's family
                    children_idxs = []
                    if hgf.edges[idx].value_children is not None:
                        children_idxs.extend(
                            [*hgf.edges[idx].value_children]  # type: ignore
                        )
                    if hgf.edges[idx].volatility_children is not None:
                        children_idxs.extend(
                            [*hgf.edges[idx].volatility_children]  # type: ignore
                        )

                    if len(children_idxs) > 0:
                        # only call the update on the parent(s) if the current node
                        # is the last on the children's list (meaning that all the
                        # other nodes have been updated)
                        if children_idxs[-1] == node_idx:
                            update_sequence, prediction_sequence = get_update_sequence(
                                hgf=hgf,
                                node_idxs=(idx,),
                                update_sequence=update_sequence,
                                prediction_sequence=prediction_sequence,
                                init=False,
                            )
    # final iteration
    if init is True:
        # add the categorical update here, if any
        for kind, idx in zip(hgf.input_nodes_idx.kind, hgf.input_nodes_idx.idx):
            if kind == "categorical":
                # create a new sequence step and add it to the list
                update_sequence.append((idx, categorical_input_update))

    return update_sequence, prediction_sequence


def to_pandas(hgf: "HGF") -> pd.DataFrame:
    """Export the nodes trajectories and surprise as a Pandas data frame.

    Returns
    -------
    structure_df :
        Pandas data frame with the time series of sufficient statistics and
        the surprise of each node in the structure.

    """
    # get time and time steps from the first input node
    structure_df = pd.DataFrame(
        {
            "time_steps": hgf.node_trajectories[hgf.input_nodes_idx.idx[0]][
                "time_step"
            ],
            "time": jnp.cumsum(
                hgf.node_trajectories[hgf.input_nodes_idx.idx[0]]["time_step"]
            ),
        }
    )

    # add the observations from input nodes
    for idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        if kind == "categorical":
            df = pd.DataFrame(
                dict(
                    [
                        (
                            f"observation_input_{idx}_{i}",
                            hgf.node_trajectories[0]["value"][:, i],
                        )
                        for i in range(hgf.node_trajectories[0]["value"].shape[1])
                    ]
                )
            )
            pd.concat([structure_df, df], axis=1)
        else:
            structure_df[f"observation_input_{idx}"] = hgf.node_trajectories[idx][
                "value"
            ]

    # loop over non input nodes and store sufficient statistics with surprise
    indexes = [
        i
        for i in range(1, len(hgf.node_trajectories))
        if i not in hgf.input_nodes_idx.idx
    ]
    df = pd.DataFrame(
        dict(
            [
                (f"x_{i}_{var}", hgf.node_trajectories[i][var])
                for i in indexes
                for var in ["mu", "pi", "muhat", "pihat"]
            ]
        )
    )
    structure_df = pd.concat([structure_df, df], axis=1)

    # for value parents of continuous inputs nodes
    continuous_parents_idxs = []
    for idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        if kind == "continuous":
            for par_idx in hgf.edges[idx].value_parents:  # type: ignore
                continuous_parents_idxs.append(par_idx)

    for i in indexes:
        if i in continuous_parents_idxs:
            surprise = gaussian_surprise(
                x=hgf.node_trajectories[0]["value"],
                muhat=hgf.node_trajectories[i]["muhat"],
                pihat=hgf.node_trajectories[i]["pihat"],
            )
        else:
            surprise = gaussian_surprise(
                x=hgf.node_trajectories[i]["mu"],
                muhat=hgf.node_trajectories[i]["muhat"],
                pihat=hgf.node_trajectories[i]["pihat"],
            )

        # fill with nans when the model cannot fit
        surprise = jnp.where(
            jnp.isnan(hgf.node_trajectories[i]["mu"]), jnp.nan, surprise
        )
        structure_df[f"x_{i}_surprise"] = surprise

    # compute the global surprise over all node
    structure_df["surprise"] = structure_df.iloc[
        :, structure_df.columns.str.contains("_surprise")
    ].sum(axis=1, min_count=1)

    return structure_df
