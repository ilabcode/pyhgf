# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.math import Normal, binary_surprise, gaussian_surprise
from pyhgf.typing import AdjacencyLists, Attributes, Edges, Sequence, UpdateSequence
from pyhgf.updates.observation import set_observation
from pyhgf.updates.posterior.categorical import categorical_state_update
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
)
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction.dirichlet import dirichlet_node_prediction
from pyhgf.updates.prediction_error.binary import binary_state_node_prediction_error
from pyhgf.updates.prediction_error.categorical import (
    categorical_state_prediction_error,
)
from pyhgf.updates.prediction_error.continuous import continuous_node_prediction_error
from pyhgf.updates.prediction_error.dirichlet import dirichlet_node_prediction_error
from pyhgf.updates.prediction_error.exponential import (
    prediction_error_update_exponential_family,
)
from pyhgf.updates.prediction_error.generic import generic_state_prediction_error

if TYPE_CHECKING:
    from pyhgf.model import Network


@partial(jit, static_argnames=("update_sequence", "edges", "input_idxs"))
def beliefs_propagation(
    attributes: Attributes,
    inputs: Tuple[ArrayLike, ...],
    update_sequence: UpdateSequence,
    edges: Edges,
    input_idxs: Tuple[int],
) -> Tuple[Dict, Dict]:
    """Update the network's parameters after observing new data point(s).

    This function performs the beliefs propagation step. Belief propagation consists in:
    1. A prediction sequence, from the leaves of the graph to the roots.
    2. The assignation of new observations to target nodes (usually the roots of the
    network)
    3. An inference step alternating between prediction errors and posterior updates,
    starting from the roots of the network to the leaves.
    This function returns a tuple of two new `parameter_structure` (i.e. the carryover
    and the accumulated in the context of :py:func:`jax.lax.scan`).

    Parameters
    ----------
    attributes :
        The dictionaries of nodes' parameters. This variable is updated and returned
        after the beliefs propagation step.
    inputs :
        A tuple of n by time steps arrays containing the new observation(s), the time
        steps as well as a boolean mask for observed values. The new observations are a
        tuple of array, with length equal to the number of input nodes. Each input node
        can receive observations  The time steps are the last
        column of the array, the default is unit incrementation.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    edges :
        Information on the network's edges.
    input_idxs :
        List input indexes.

    Returns
    -------
    attributes, attributes :
        A tuple of parameters structure (carryover and accumulated).

    """
    prediction_steps, update_steps = update_sequence

    # unpack input data - input_values is a tuple of n x time steps arrays
    (*input_data, time_step) = inputs

    attributes[-1]["time_step"] = time_step

    # Prediction sequence
    # -------------------
    for step in prediction_steps:

        node_idx, update_fn = step

        attributes = update_fn(
            attributes=attributes,
            node_idx=node_idx,
            edges=edges,
        )

    # Observations
    # ------------
    for values, observed, node_idx in zip(
        input_data[::2], input_data[1::2], input_idxs
    ):

        attributes = set_observation(
            attributes=attributes,
            node_idx=node_idx,
            values=values,
            observed=observed,
        )

    # Update sequence
    # ---------------
    for step in update_steps:

        node_idx, update_fn = step

        attributes = update_fn(
            attributes=attributes,
            node_idx=node_idx,
            edges=edges,
        )

    return (
        attributes,
        attributes,
    )  # ("carryover", "accumulated")


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
    binary_states_idxs: List[int],
    binary_parameters: Dict,
) -> "Network":
    """Generate a binary network implied by categorical state(-transition) nodes.

    Parameters
    ----------
    network :
        Instance of a Network.
    node_idx :
        Index to the categorical state node.
    binary_states_idxs :
        The indexes of the binary state nodes.
    binary_parameters :
        Parameters for the set of implied binary HGFs.

    Returns
    -------
    hgf :
        The updated instance of the HGF model.

    """
    # add the binary states - one for each category
    network.add_nodes(
        kind="binary-state",
        n_nodes=len(binary_states_idxs),
        node_parameters={
            "mean": binary_parameters["mean_1"],
            "precision": binary_parameters["precision_1"],
        },
    )

    # add the value coupling between the categorical and binary states
    edges_as_list: List[AdjacencyLists] = list(network.edges)
    edges_as_list[node_idx] = AdjacencyLists(
        5, tuple(binary_states_idxs), None, None, None, (None,)
    )
    for binary_idx in binary_states_idxs:
        edges_as_list[binary_idx] = AdjacencyLists(
            1, None, None, (node_idx,), None, (None,)
        )
    network.edges = tuple(edges_as_list)

    # add continuous state parent nodes
    n_nodes = len(network.edges)
    for i in range(binary_parameters["n_categories"]):
        network.add_nodes(
            value_children=i + n_nodes - binary_parameters["n_categories"],
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
            idx + binary_parameters["n_categories"] for idx in binary_states_idxs
        ],
        node_parameters={
            "mean": binary_parameters["mean_3"],
            "precision": binary_parameters["precision_3"],
            "tonic_volatility": binary_parameters["tonic_volatility_3"],
        },
    )

    return network


def get_update_sequence(
    network: "Network", update_type: str
) -> Tuple[Sequence, Sequence]:
    """Generate an update sequence from the network's structure.

    This function return an optimized update sequence considering the edges of the
    network. The function ensures that the following principles apply:
    1. all children have computed prediction errors before the parent is updated.
    2. all children have been updated before the parent compute the prediction errors.

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
    prediction_sequence :
        The sequence of prediction update.
    update_sequence :
        The sequence of prediction error and posterior updates.

    """
    # initialize the update and prediction sequences
    update_sequence: List = []
    prediction_sequence: List = []

    n_nodes = len(network.edges)

    # list all nodes that are not triggering prediction errors or posterior updates
    # do not call posterior updates for nodes without children (input nodes)
    nodes_without_prediction_error = [i for i in range(n_nodes)]
    nodes_without_prediction = [i for i in range(n_nodes)]
    nodes_without_posterior_update = [
        i
        for i in range(n_nodes)
        if not (
            (network.edges[i].value_children is None)
            & (network.edges[i].volatility_children is None)
        )
    ]

    # prediction updates ---------------------------------------------------------------
    while True:
        no_update = True

        # for all nodes that should apply prediction update ----------------------------
        # verify that all children have computed the prediction error
        for idx in nodes_without_prediction:
            all_parents = [
                i
                for idx in [
                    network.edges[idx].value_parents,
                    network.edges[idx].volatility_parents,
                ]
                if idx is not None
                for i in idx
            ]

            # there is no parent waiting for a prediction update
            if not any([i in nodes_without_prediction for i in all_parents]):
                no_update = False
                nodes_without_prediction.remove(idx)
                if network.edges[idx].node_type == 1:
                    prediction_sequence.append((idx, binary_state_node_prediction))
                elif network.edges[idx].node_type == 2:
                    prediction_sequence.append((idx, continuous_node_prediction))
                elif network.edges[idx].node_type == 4:
                    prediction_sequence.append((idx, dirichlet_node_prediction))

        if not nodes_without_prediction:
            break

        if no_update:
            raise Warning(
                "The structure of the network cannot be updated consistently."
            )

    # prediction errors and posterior updates
    # will fail if the structure of the network does not allow a consistent update order
    # ----------------------------------------------------------------------------------
    while True:
        no_update = True

        # for all nodes that should apply posterior update -----------------------------
        # verify that all children have computed the prediction error
        update_fn = None
        for idx in nodes_without_posterior_update:
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
            if all([i not in nodes_without_prediction_error for i in all_children]):
                no_update = False
                if network.edges[idx].node_type == 2:
                    if update_type == "eHGF":
                        if network.edges[idx].volatility_children is not None:
                            update_fn = continuous_node_posterior_update_ehgf
                        else:
                            update_fn = continuous_node_posterior_update
                    elif update_type == "standard":
                        update_fn = continuous_node_posterior_update

                elif network.edges[idx].node_type == 4:

                    update_fn = None

                update_sequence.append((idx, update_fn))
                nodes_without_posterior_update.remove(idx)

        # for all nodes that should apply prediction error------------------------------
        # verify that all children have been updated
        update_fn = None
        for idx in nodes_without_prediction_error:

            all_parents = [
                i
                for idx in [
                    network.edges[idx].value_parents,
                    network.edges[idx].volatility_parents,
                ]
                if idx is not None
                for i in idx
            ]

            # if this node has no parent, no need to compute prediction errors
            # unless this is an exponential family state node
            if len(all_parents) == 0:
                if network.edges[idx].node_type == 3:
                    # create the sufficient statistic function
                    # for the exponential family node
                    ef_update = Partial(
                        prediction_error_update_exponential_family,
                        sufficient_stats_fn=Normal().sufficient_statistics,
                    )
                    update_fn = ef_update
                    no_update = False
                    update_sequence.append((idx, update_fn))
                    nodes_without_prediction_error.remove(idx)
                else:
                    nodes_without_prediction_error.remove(idx)
            else:
                # if this node has been updated
                if idx not in nodes_without_posterior_update:

                    if network.edges[idx].node_type == 0:
                        update_fn = generic_state_prediction_error
                    elif network.edges[idx].node_type == 1:
                        update_fn = binary_state_node_prediction_error
                    elif network.edges[idx].node_type == 2:
                        update_fn = continuous_node_prediction_error
                    elif network.edges[idx].node_type == 4:
                        update_fn = dirichlet_node_prediction_error
                    elif network.edges[idx].node_type == 5:
                        update_fn = categorical_state_prediction_error

                        # add the update here, this will move at the end of the sequence
                        update_sequence.append((idx, categorical_state_update))
                    else:
                        raise ValueError(f"Invalid node type encountered at node {idx}")

                    no_update = False
                    update_sequence.append((idx, update_fn))
                    nodes_without_prediction_error.remove(idx)

        if (not nodes_without_prediction_error) and (
            not nodes_without_posterior_update
        ):
            break

        if no_update:
            raise Warning(
                "The structure of the network cannot be updated consistently."
            )

    # remove None steps and return the update sequence
    prediction_sequence = [
        update for update in prediction_sequence if update[1] is not None
    ]
    update_sequence = [update for update in update_sequence if update[1] is not None]

    # move all categorical steps at the end of the sequence
    for step in update_sequence:
        if not isinstance(step[1], Partial):
            if step[1].__name__ == "categorical_state_update":
                update_sequence.remove(step)
                update_sequence.append(step)

    return tuple(prediction_sequence), tuple(update_sequence)


def to_pandas(network: "Network") -> pd.DataFrame:
    """Export the nodes trajectories and surprise as a Pandas data frame.

    Returns
    -------
    trajectories_df :
        Pandas data frame with the time series of sufficient statistics and the
        surprise of each node in the structure.

    """
    n_nodes = len(network.edges)
    # get time and time steps from the first input node
    trajectories_df = pd.DataFrame(
        {
            "time_steps": network.node_trajectories[-1]["time_step"],
            "time": jnp.cumsum(network.node_trajectories[-1]["time_step"]),
        }
    )

    # loop over continuous and binary state nodes and store sufficient statistics
    # ---------------------------------------------------------------------------
    states_indexes = [i for i in range(n_nodes) if network.edges[i].node_type in [1, 2]]
    df = pd.DataFrame(
        dict(
            [
                (f"x_{i}_{var}", network.node_trajectories[i][var])
                for i in states_indexes
                for var in network.node_trajectories[i].keys()
                if (("mean" in var) or ("precision" in var))
            ]
        )
    )
    trajectories_df = pd.concat([trajectories_df, df], axis=1)

    # loop over exponential family state nodes and store sufficient statistics
    # ------------------------------------------------------------------------
    ef_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 3]
    for i in ef_indexes:
        for var in ["nus", "xis", "mean"]:
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

    # add surprise from binary state nodes
    binary_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 1]
    for bin_idx in binary_indexes:
        surprise = binary_surprise(
            x=network.node_trajectories[bin_idx]["mean"],
            expected_mean=network.node_trajectories[bin_idx]["expected_mean"],
        )
        trajectories_df[f"x_{bin_idx}_surprise"] = surprise

    # add surprise from continuous state nodes
    continuous_indexes = [i for i in range(n_nodes) if network.edges[i].node_type == 2]
    for con_idx in continuous_indexes:
        surprise = gaussian_surprise(
            x=network.node_trajectories[con_idx]["mean"],
            expected_mean=network.node_trajectories[con_idx]["expected_mean"],
            expected_precision=network.node_trajectories[con_idx]["expected_precision"],
        )
        trajectories_df[f"x_{con_idx}_surprise"] = surprise

    # compute the global surprise over all node
    trajectories_df["total_surprise"] = trajectories_df.iloc[
        :, trajectories_df.columns.str.contains("_surprise")
    ].sum(axis=1, min_count=1)

    return trajectories_df


def add_edges(
    attributes: Dict,
    edges: Edges,
    kind="value",
    parent_idxs=Union[int, List[int]],
    children_idxs=Union[int, List[int]],
    coupling_strengths: Union[float, List[float], Tuple[float]] = 1.0,
    coupling_fn: Tuple[Optional[Callable], ...] = (None,),
) -> Tuple:
    """Add a value or volatility coupling link between a set of nodes.

    Parameters
    ----------
    attributes :
        Attributes of the neural network.
    edges :
        Edges of the neural network.
    kind :
        The kind of coupling can be `"value"` or `"volatility"`.
    parent_idxs :
        The index(es) of the parent node(s).
    children_idxs :
        The index(es) of the children node(s).
    coupling_strengths :
        The coupling strength between the parents and children.
    coupling_fn :
        Coupling function(s) between the current node and its value children.
        It has to be provided as a tuple. If multiple value children are specified,
        the coupling functions must be stated in the same order of the children.
        Note: if a node has multiple parents nodes with different coupling
        functions, a coupling function should be indicated for all the parent nodes.
        If no coupling function is stated, the relationship between nodes is assumed
        linear.

    """
    if kind not in ["value", "volatility"]:
        raise ValueError(
            f"The kind of coupling should be value or volatility, got {kind}"
        )
    if isinstance(children_idxs, int):
        children_idxs = [children_idxs]
    assert isinstance(children_idxs, (list, tuple))

    if isinstance(parent_idxs, int):
        parent_idxs = [parent_idxs]
    assert isinstance(parent_idxs, (list, tuple))

    if isinstance(coupling_strengths, int):
        coupling_strengths = [float(coupling_strengths)]
    if isinstance(coupling_strengths, float):
        coupling_strengths = [coupling_strengths]

    assert isinstance(coupling_strengths, (list, tuple))

    edges_as_list = list(edges)
    # update the parent nodes
    # -----------------------
    for parent_idx in parent_idxs:
        # unpack the parent's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            this_coupling_fn,
        ) = edges_as_list[parent_idx]

        if kind == "value":
            if value_children is None:
                value_children = tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] = tuple(
                    coupling_strengths
                )
            else:
                value_children = value_children + tuple(children_idxs)
                attributes[parent_idx]["value_coupling_children"] += tuple(
                    coupling_strengths
                )
                this_coupling_fn = this_coupling_fn + coupling_fn
        elif kind == "volatility":
            if volatility_children is None:
                volatility_children = tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] = tuple(
                    coupling_strengths
                )
            else:
                volatility_children = volatility_children + tuple(children_idxs)
                attributes[parent_idx]["volatility_coupling_children"] += tuple(
                    coupling_strengths
                )

        # save the updated edges back
        edges_as_list[parent_idx] = AdjacencyLists(
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            this_coupling_fn,
        )

    # update the children nodes
    # -------------------------
    for children_idx in children_idxs:
        # unpack this node's edges
        (
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            coupling_fn,
        ) = edges_as_list[children_idx]

        if kind == "value":
            if value_parents is None:
                value_parents = tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] = tuple(
                    coupling_strengths
                )
            else:
                value_parents = value_parents + tuple(parent_idxs)
                attributes[children_idx]["value_coupling_parents"] += tuple(
                    coupling_strengths
                )
        elif kind == "volatility":
            if volatility_parents is None:
                volatility_parents = tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] = tuple(
                    coupling_strengths
                )
            else:
                volatility_parents = volatility_parents + tuple(parent_idxs)
                attributes[children_idx]["volatility_coupling_parents"] += tuple(
                    coupling_strengths
                )

        # save the updated edges back
        edges_as_list[children_idx] = AdjacencyLists(
            node_type,
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
            coupling_fn,
        )

    # convert the list back to a tuple
    edges = tuple(edges_as_list)

    return attributes, edges


def get_input_idxs(edges: Edges) -> Tuple[int, ...]:
    """List all possible default inputs nodes.

    An input node is a state node without any child.

    Parameters
    ----------
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.

    """
    return tuple(
        [
            i
            for i in range(len(edges))
            if (
                (edges[i].value_children is None)
                & (edges[i].volatility_children is None)
            )
        ]
    )
