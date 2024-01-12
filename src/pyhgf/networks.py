# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import TYPE_CHECKING, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise, gaussian_surprise
from pyhgf.typing import Indexes, UpdateSequence
from pyhgf.updates.posterior.binary import binary_node_update_infinite
from pyhgf.updates.posterior.categorical import categorical_input_update
from pyhgf.updates.posterior.continuous import (
    continuous_node_update,
    continuous_node_update_ehgf,
)
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction_error.inputs.binary import (
    binary_input_prediction_error_infinite_precision,
)
from pyhgf.updates.prediction_error.inputs.continuous import (
    continuous_input_prediction_error,
)
from pyhgf.updates.prediction_error.nodes.binary import (
    binary_state_node_prediction_error,
)
from pyhgf.updates.prediction_error.nodes.continuous import (
    continuous_node_prediction_error,
)

if TYPE_CHECKING:
    from pyhgf.model import HGF


@partial(jit, static_argnames=("update_sequence", "edges", "input_nodes_idx"))
def beliefs_propagation(
    attributes: Dict,
    input_data: ArrayLike,
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
    input_data :
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
    # extract value(s) and time steps
    values, time_step, observed = input_data

    input_nodes_idx = jnp.asarray(input_nodes_idx)
    # Fit the model with the current time and value variables, given the model structure
    for sequence in update_sequence:
        node_idx, update_fn = sequence

        # if we are updating an input node, select the value that should be passed
        # otherwise, just pass 0.0 and the value will be ignored
        value = jnp.sum(jnp.where(jnp.equal(input_nodes_idx, node_idx), values, 0.0))

        # is it an observation or a missing observation
        observed_value = jnp.sum(jnp.equal(input_nodes_idx, node_idx) * observed)

        attributes = update_fn(
            attributes=attributes,
            time_step=time_step[0],
            node_idx=node_idx,
            edges=edges,
            value=value,
            observed=observed_value,
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
    hgf: "HGF", node_idx: int, binary_input_idxs: List[int], binary_parameters: Dict
) -> "HGF":
    """Generate a binary network implied by categorical state(-transition) nodes.

    Parameters
    ----------
    hgf :
        Instance of a HGF model.
    node_idx :
        Index to the categorical state node.
    binary_input_idxs :
        The idexes of the binary input nodes.
    binary_parameters :
        Parameters for the set of implied binary HGFs.

    Returns
    -------
    hgf :
        The updated instance of the HGF model.

    """
    # add the binary inputs - one for each category
    hgf.add_nodes(
        kind="binary-input",
        n_nodes=len(binary_input_idxs),
        node_parameters={
            key: binary_parameters[key] for key in ["eta0", "eta1", "binary_precision"]
        },
    )

    # add the value dependency between the categorical and binary nodes
    edges_as_list: List[Indexes] = list(hgf.edges)
    edges_as_list[node_idx] = Indexes(tuple(binary_input_idxs), None, None, None)
    for binary_idx in binary_input_idxs:
        edges_as_list[binary_idx] = Indexes(None, None, (node_idx,), None)
    hgf.edges = tuple(edges_as_list)

    # loop over the number of categories and create as many second-levels binary HGF
    for i in range(binary_parameters["n_categories"]):
        # binary state node
        hgf.add_nodes(
            kind="binary-state",
            value_children=binary_input_idxs[i],
            node_parameters={
                "mean": binary_parameters["mean_1"],
                "precision": binary_parameters["precision_1"],
            },
        )

    # add the continuous parent node
    for i in range(binary_parameters["n_categories"]):
        hgf.add_nodes(
            value_children=binary_input_idxs[i] + binary_parameters["n_categories"],
            node_parameters={
                "mean": binary_parameters["mean_2"],
                "precision": binary_parameters["precision_2"],
                "tonic_volatility": binary_parameters["tonic_volatility_2"],
            },
        )

    # add the higher level volatility parents
    # as a shared parents between the second level nodes
    hgf.add_nodes(
        volatility_children=[
            idx + 2 * binary_parameters["n_categories"] for idx in binary_input_idxs
        ],
        node_parameters={
            "mean": binary_parameters["mean_3"],
            "precision": binary_parameters["precision_3"],
            "tonic_volatility": binary_parameters["tonic_volatility_3"],
        },
    )

    return hgf


def get_update_sequence(hgf: "HGF") -> List:
    """Generate an update sequence from the network's structure.

    This function return an optimized update sequence considering the edges of the
    network. The function ensures that the following principles apply:
    1. all children have computed prediction errors before the parent is updated.
    2. all children have been updated before the parent compute the prediction errors.
    3. the update function of an input node is chosen based on the node's type
    (`"continuous"`, `"binary"` or `"categorical"`).
    4. the update function of the parent of an input node is chosen based on the
    node's type (`"continuous"`, `"binary"` or `"categorical"`).

    Parameters
    ----------
    hgf :
        An instance of the HGF model.

    Returns
    -------
    sequence :
        The update sequence generated from the node structure.

    """
    # initialize sequences
    update_sequence: List = []
    prediction_sequence: List = []

    n_nodes = len(hgf.edges)

    # list all nodes that should compute PE and update
    node_without_pe = [i for i in range(n_nodes)]
    node_without_update = [i for i in range(n_nodes)]

    # start with the update of all input nodes
    for input_idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        if kind == "continuous":
            update_fn = continuous_input_prediction_error
            update_sequence.append((input_idx, update_fn))

        elif kind == "binary":
            # add the update steps for the binary state node as well
            binary_state_idx = hgf.edges[input_idx].value_parents[0]  # type: ignore

            # TODO: once HGF and Network classes implemented separately, we can add the
            # finite binary HGF back
            # if hgf.attributes[input_idx]["expected_precision"] == jnp.inf:
            update_sequence.append(
                (input_idx, binary_input_prediction_error_infinite_precision)
            )
            update_sequence.append((binary_state_idx, binary_node_update_infinite))
            # else:
            #     update_sequence.append(
            #         (input_idx, binary_input_prediction_error_finite_precision)
            #     )
            #     update_sequence.append((binary_state_idx, binary_node_update_finite))
            update_sequence.append(
                (binary_state_idx, binary_state_node_prediction_error)
            )
            node_without_pe.remove(binary_state_idx)
            node_without_update.remove(binary_state_idx)
            prediction_sequence.insert(
                0, (binary_state_idx, binary_state_node_prediction)
            )

        # add the PE step to the sequence
        node_without_pe.remove(input_idx)

        # input node does not need to update the posterior
        node_without_update.remove(input_idx)

    # will fail if the structure of the network does not allow a consistent update order
    while True:
        no_update = True

        # for all nodes that should be updated -----------------------------------------
        # verify that all children have been computed the PE
        for idx in node_without_update:
            all_children = [
                i
                for idx in [
                    hgf.edges[idx].value_children,
                    hgf.edges[idx].volatility_children,
                ]
                if idx is not None
                for i in idx
            ]

            # all the children have computed prediction errors
            if all([i not in node_without_pe for i in all_children]):
                no_update = False

                if hgf.update_type == "eHGF":
                    if hgf.edges[idx].volatility_children is not None:
                        update_fn = continuous_node_update_ehgf
                    else:
                        update_fn = continuous_node_update
                elif hgf.update_type == "standard":
                    update_fn = continuous_node_update

                update_sequence.append((idx, update_fn))
                node_without_update.remove(idx)

                # the prediction sequence is the update sequence in reverse order
                prediction_sequence.insert(0, (idx, continuous_node_prediction))

        # for all nodes that should compute a PE ---------------------------------------
        # verify that all children have been updated and compute the PE
        for idx in node_without_pe:
            # if this node has no parent, no need to compute prediction errors
            all_parents = [
                i
                for idx in [
                    hgf.edges[idx].value_parents,
                    hgf.edges[idx].volatility_parents,
                ]
                if idx is not None
                for i in idx
            ]
            if len(all_parents) == 0:
                node_without_pe.remove(idx)
            else:
                # if this node has been updated
                if idx not in node_without_update:
                    no_update = False
                    update_sequence.append((idx, continuous_node_prediction_error))
                    node_without_pe.remove(idx)

        if (not node_without_pe) and (not node_without_update):
            break

        if no_update:
            break
            raise Warning(
                "The structure of the network cannot be updated consistently."
            )

    # append the prediction sequence at the beginning
    prediction_sequence.extend(update_sequence)

    # add the categorical update here, if any
    for kind, idx in zip(hgf.input_nodes_idx.kind, hgf.input_nodes_idx.idx):
        if kind == "categorical":
            # create a new sequence step and add it to the list
            prediction_sequence.append((idx, categorical_input_update))

    return prediction_sequence


def to_pandas(hgf: "HGF") -> pd.DataFrame:
    """Export the nodes trajectories and surprise as a Pandas data frame.

    Returns
    -------
    trajectories_df :
        Pandas data frame with the time series of sufficient statistics and
        the surprise of each node in the structure.

    """
    # get time and time steps from the first input node
    trajectories_df = pd.DataFrame(
        {
            "time_steps": hgf.node_trajectories[hgf.input_nodes_idx.idx[0]][
                "time_step"
            ],
            "time": jnp.cumsum(
                hgf.node_trajectories[hgf.input_nodes_idx.idx[0]]["time_step"]
            ),
        }
    )

    # Observations
    # ------------

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
            pd.concat([trajectories_df, df], axis=1)
        else:
            trajectories_df[f"observation_input_{idx}"] = hgf.node_trajectories[idx][
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
                for var in ["mean", "precision", "expected_mean", "expected_precision"]
            ]
        )
    )
    trajectories_df = pd.concat([trajectories_df, df], axis=1)

    # Surprises
    # ---------

    # add the surprise computed from the continuous inputs
    for idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        trajectories_df[f"observation_input_{idx}_surprise"] = hgf.node_trajectories[
            idx
        ]["surprise"]

    # for value parents of binary inputs nodes
    binary_nodes_idxs = []
    for idx, kind in zip(hgf.input_nodes_idx.idx, hgf.input_nodes_idx.kind):
        if kind == "binary":
            for par_idx in hgf.edges[idx].value_parents:  # type: ignore
                binary_nodes_idxs.append(par_idx)

    for i in indexes:
        if i in binary_nodes_idxs:
            binary_input = hgf.edges[i].value_children[0]  # type: ignore
            surprise = binary_surprise(
                x=hgf.node_trajectories[binary_input]["value"],
                expected_mean=hgf.node_trajectories[i]["expected_mean"],
            )
        else:
            surprise = gaussian_surprise(
                x=hgf.node_trajectories[i]["mean"],
                expected_mean=hgf.node_trajectories[i]["expected_mean"],
                expected_precision=hgf.node_trajectories[i]["expected_precision"],
            )

        # fill with nans when the model cannot fit
        surprise = jnp.where(
            jnp.isnan(hgf.node_trajectories[i]["mean"]), jnp.nan, surprise
        )
        trajectories_df[f"x_{i}_surprise"] = surprise

    # compute the global surprise over all node
    trajectories_df["total_surprise"] = trajectories_df.iloc[
        :, trajectories_df.columns.str.contains("_surprise")
    ].sum(axis=1, min_count=1)

    return trajectories_df
