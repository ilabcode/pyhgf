# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import TYPE_CHECKING, Dict, List, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.math import Normal, binary_surprise, gaussian_surprise
from pyhgf.typing import AdjacencyLists, Attributes, Structure, UpdateSequence
from pyhgf.updates.posterior.binary import binary_node_update_infinite
from pyhgf.updates.posterior.categorical import categorical_input_update
from pyhgf.updates.posterior.continuous import (
    continuous_node_update,
    continuous_node_update_ehgf,
)
from pyhgf.updates.posterior.exponential import posterior_update_exponential_family
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction_error.inputs.binary import (
    binary_input_prediction_error_infinite_precision,
)
from pyhgf.updates.prediction_error.inputs.continuous import (
    continuous_input_prediction_error,
)
from pyhgf.updates.prediction_error.inputs.generic import generic_input_prediction_error
from pyhgf.updates.prediction_error.nodes.binary import (
    binary_state_node_prediction_error,
)
from pyhgf.updates.prediction_error.nodes.continuous import (
    continuous_node_prediction_error,
)

if TYPE_CHECKING:
    from pyhgf.model import Network


@partial(jit, static_argnames=("update_sequence", "structure"))
def beliefs_propagation(
    attributes: Attributes,
    input_data: ArrayLike,
    update_sequence: UpdateSequence,
    structure: Structure,
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
        matching the length `inputs.idx`. `inputs.idx` is used to index the
        ith value in the vector to the ith input node, so the ordering of the input
        array matters. The time steps are the last column of the array, the default is
        unit incrementation.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    structure :
        Information on the network's structure, including input types and edges.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    """
    # unpack static info about the network structure
    inputs, edges = structure
    input_idxs = jnp.array(inputs.idx)

    # extract value(s) and time steps
    values, time_step, observed = input_data

    # Fit the model with the current time and value variables, given the model structure
    for sequence in update_sequence:
        node_idx, update_fn = sequence

        # if we are updating an input node, select the value that should be passed
        # otherwise, just pass 0.0 and the value will be ignored
        value = jnp.sum(jnp.where(jnp.equal(input_idxs, node_idx), values, 0.0))

        # is it an observation or a missing observation
        observed_value = jnp.sum(jnp.equal(input_idxs, node_idx) * observed)

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
    network: "Network",
    node_idx: int,
    binary_input_idxs: List[int],
    binary_parameters: Dict,
) -> "Network":
    """Generate a binary network implied by categorical state(-transition) nodes.

    Parameters
    ----------
    network :
        Instance of a Network.
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
    network.add_nodes(
        kind="binary-input",
        n_nodes=len(binary_input_idxs),
        node_parameters={
            key: binary_parameters[key] for key in ["eta0", "eta1", "binary_precision"]
        },
    )

    # add the value dependency between the categorical and binary nodes
    edges_as_list: List[AdjacencyLists] = list(network.edges)
    edges_as_list[node_idx] = AdjacencyLists(
        0, tuple(binary_input_idxs), None, None, None
    )
    for binary_idx in binary_input_idxs:
        edges_as_list[binary_idx] = AdjacencyLists(0, None, None, (node_idx,), None)
    network.edges = tuple(edges_as_list)

    # loop over the number of categories and create as many second-levels binary HGF
    for i in range(binary_parameters["n_categories"]):
        # binary state node
        network.add_nodes(
            kind="binary-state",
            value_children=binary_input_idxs[i],
            node_parameters={
                "mean": binary_parameters["mean_1"],
                "precision": binary_parameters["precision_1"],
            },
        )

    # add the continuous parent node
    for i in range(binary_parameters["n_categories"]):
        network.add_nodes(
            value_children=binary_input_idxs[i] + binary_parameters["n_categories"],
            node_parameters={
                "mean": binary_parameters["mean_2"],
                "precision": binary_parameters["precision_2"],
                "tonic_volatility": binary_parameters["tonic_volatility_2"],
            },
        )

    # add the higher level volatility parents
    # as a shared parents between the second level nodes
    network.add_nodes(
        volatility_children=[
            idx + 2 * binary_parameters["n_categories"] for idx in binary_input_idxs
        ],
        node_parameters={
            "mean": binary_parameters["mean_3"],
            "precision": binary_parameters["precision_3"],
            "tonic_volatility": binary_parameters["tonic_volatility_3"],
        },
    )

    return network


def get_update_sequence(network: "Network", update_type: str) -> List:
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
    network :
        A neural network, instance of :py:class:`pyhgf.model.network.Network`.
    update_type :
        The type of update to perform for volatility coupling. Can be `"eHGF"`
        (defaults) or `"standard"`. The eHGF update step was proposed as an
        alternative to the original definition in that it starts by updating the
        mean and then the precision of the parent node, which generally reduces the
        errors associated with impossible parameter space and improves sampling.

    Returns
    -------
    sequence :
        The update sequence generated from the node structure.

    """
    # initialize the update and prediction sequences
    update_sequence: List = []
    prediction_sequence: List = []

    n_nodes = len(network.edges)

    # list all nodes that are not triggering prediction errors or posterior updates
    node_without_pe = [i for i in range(n_nodes)]
    node_without_update = [i for i in range(n_nodes)]

    # start by injecting the observations in all input nodes
    for input_idx, kind in zip(network.inputs.idx, network.inputs.kind):
        if kind == 0:
            update_fn = continuous_input_prediction_error
            update_sequence.append((input_idx, update_fn))

        elif kind == 1:
            # add the update steps for the binary state node as well
            binary_state_idx = network.edges[input_idx].value_parents[0]  # type: ignore

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

        elif kind == 3:
            update_fn = generic_input_prediction_error
            update_sequence.append((input_idx, update_fn))

        # add the PE step to the sequence
        node_without_pe.remove(input_idx)

        # input node does not need to update the posterior
        node_without_update.remove(input_idx)

    # will fail if the structure of the network does not allow a consistent update order
    while True:
        no_update = True

        # for all nodes that should be updated
        # verify that all children have been computed the PE
        for idx in node_without_update:
            all_children = [
                i
                for idx in [
                    network.edges[idx].value_children,
                    network.edges[idx].volatility_children,
                ]
                if idx is not None
                for i in idx
            ]

            # all the children have computed prediction errors
            if all([i not in node_without_pe for i in all_children]):
                no_update = False
                if network.edges[idx].node_type == 2:
                    if update_type == "eHGF":
                        if network.edges[idx].volatility_children is not None:
                            update_fn = continuous_node_update_ehgf
                        else:
                            update_fn = continuous_node_update
                    elif update_type == "standard":
                        update_fn = continuous_node_update

                    # the prediction sequence is the update sequence in reverse order
                    prediction_sequence.insert(0, (idx, continuous_node_prediction))

                elif network.edges[idx].node_type == 3:

                    # create the sufficient statistic function
                    # for the exponential family node
                    ef_update = Partial(
                        posterior_update_exponential_family,
                        sufficient_stats_fn=Normal.sufficient_statistics,
                    )
                    update_fn = ef_update

                update_sequence.append((idx, update_fn))
                node_without_update.remove(idx)

        # for all nodes that should compute a PE ---------------------------------------
        # verify that all children have been updated and compute the PE
        for idx in node_without_pe:
            # if this node has no parent, no need to compute prediction errors
            all_parents = [
                i
                for idx in [
                    network.edges[idx].value_parents,
                    network.edges[idx].volatility_parents,
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
    for kind, idx in zip(network.inputs.kind, network.inputs.idx):
        if kind == 2:
            # create a new sequence step and add it to the list
            prediction_sequence.append((idx, categorical_input_update))

    return prediction_sequence


def to_pandas(network: "Network") -> pd.DataFrame:
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
            "time_steps": network.node_trajectories[network.inputs.idx[0]]["time_step"],
            "time": jnp.cumsum(
                network.node_trajectories[network.inputs.idx[0]]["time_step"]
            ),
        }
    )

    # Observations
    # ------------

    # add the observations from input nodes
    for idx, kind in zip(network.inputs.idx, network.inputs.kind):
        if kind == 2:
            df = pd.DataFrame(
                dict(
                    [
                        (
                            f"observation_input_{idx}_{i}",
                            network.node_trajectories[0]["values"][:, i],
                        )
                        for i in range(network.node_trajectories[0]["values"].shape[1])
                    ]
                )
            )
            pd.concat([trajectories_df, df], axis=1)
        else:
            trajectories_df[f"observation_input_{idx}"] = network.node_trajectories[
                idx
            ]["values"]

    # loop over continuous and binary state nodes and store sufficient statistics
    # ---------------------------------------------------------------------------
    continuous_indexes = [
        i
        for i in range(1, len(network.node_trajectories))
        if network.edges[i].node_type in [1, 2]
    ]
    df = pd.DataFrame(
        dict(
            [
                (f"x_{i}_{var}", network.node_trajectories[i][var])
                for i in continuous_indexes
                for var in ["mean", "precision", "expected_mean", "expected_precision"]
            ]
        )
    )
    trajectories_df = pd.concat([trajectories_df, df], axis=1)

    # loop over exponential family state nodes and store sufficient statistics
    # ------------------------------------------------------------------------
    ef_indexes = [
        i
        for i in range(1, len(network.node_trajectories))
        if network.edges[i].node_type == 3
    ]
    for i in ef_indexes:
        for var in ["nus", "xis", "values"]:
            if network.node_trajectories[i][var].ndim == 1:
                trajectories_df = pd.concat(
                    [
                        trajectories_df,
                        pd.DataFrame(
                            dict([(f"x_{i}_{var}", network.node_trajectories[i][var])])
                        ),
                    ],
                    axis=1,
                )
            else:
                for ii in range(network.node_trajectories[i][var].shape[1]):
                    trajectories_df = pd.concat(
                        [
                            trajectories_df,
                            pd.DataFrame(
                                dict(
                                    [
                                        (
                                            f"x_{i}_{var}_{ii}",
                                            network.node_trajectories[i][var][:, ii],
                                        )
                                    ]
                                )
                            ),
                        ],
                        axis=1,
                    )

    # Expectations and surprises
    # --------------------------

    # add surprise and expected precision from continuous and binary inputs
    for idx, kind in zip(network.inputs.idx, network.inputs.kind):
        if kind in [0, 1]:
            trajectories_df[f"observation_input_{idx}_surprise"] = (
                network.node_trajectories[idx]["surprise"]
            )
            trajectories_df[f"observation_input_{idx}_expected_precision"] = (
                network.node_trajectories[idx]["expected_precision"]
            )

    # add surprise from binary state nodes
    binary_indexes = [
        i
        for i in range(1, len(network.node_trajectories))
        if network.edges[i].node_type == 1
    ]
    for i in binary_indexes:
        binary_input = network.edges[i].value_children[0]  # type: ignore
        surprise = binary_surprise(
            x=network.node_trajectories[binary_input]["values"],
            expected_mean=network.node_trajectories[i]["expected_mean"],
        )
        # fill with nans when the model cannot fit
        surprise = jnp.where(
            jnp.isnan(network.node_trajectories[i]["mean"]), jnp.nan, surprise
        )
        trajectories_df[f"x_{i}_surprise"] = surprise

    # add surprise from continuous state nodes
    for i in continuous_indexes:
        surprise = gaussian_surprise(
            x=network.node_trajectories[i]["mean"],
            expected_mean=network.node_trajectories[i]["expected_mean"],
            expected_precision=network.node_trajectories[i]["expected_precision"],
        )
        # fill with nans when the model cannot fit
        surprise = jnp.where(
            jnp.isnan(network.node_trajectories[i]["mean"]), jnp.nan, surprise
        )
        trajectories_df[f"x_{i}_surprise"] = surprise

    # compute the global surprise over all node
    trajectories_df["total_surprise"] = trajectories_df.iloc[
        :, trajectories_df.columns.str.contains("_surprise")
    ].sum(axis=1, min_count=1)

    return trajectories_df
