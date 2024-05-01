# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import copy
from functools import partial
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from jax.typing import ArrayLike
from scipy.optimize import minimize
from scipy.stats import norm

from pyhgf.typing import DirichletNode, Edges, InputIndexes

if TYPE_CHECKING:
    from pyhgf.model import HGF


def dirichlet_input_prediction_error(
    edges: Edges,
    attributes: Dict,
    value: float,
    node_idx: int,
    dirichlet_node: DirichletNode,
    input_nodes_idx,
    **args,
) -> Tuple:
    """Prediction error and update the child networks of a Dirichlet process node.

    When receiving a new input, this node chose to either:
    1. Allocate the value to a pre-existing cluster.
    2. Create a new cluster.

    The network always contains a temporary branch as the new cluster candidate. This
    branch is parametrized under the new observation to assess its likelihood and the
    previous clusters' likelihood.

    Parameters
    ----------
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    attributes :
        The attributes of the probabilistic nodes.
    value :
        The new observed value(s). The input shape should match the input shape
        at the child level.
    node_idx :
        Pointer to the Dirichlet process input node.
    dirichlet_node :
        Static parameters of the Dirichlet process node.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.
    dirichlet_node :
        Static parameters of the Dirichlet process node.

    """
    print(f"Step 1 - {attributes[1]['expected_precision']}")

    # unpack static parameters from the Dirichlet node
    (
        base_network,
        log_likelihood_fn,
        cluster_input_idxs,
    ) = dirichlet_node

    value = value  # the input value
    alpha = attributes[node_idx]["alpha"]  # the concentration parameter

    # likelihood of the current observation under possible clusters
    # -------------------------------------------------------------

    # the temporary cluster is always the last created
    temp_idx = cluster_input_idxs[-1]

    # parametrize the temporary branch given the current observed value
    # this will get a cluster proposal given the new observation
    if len(cluster_input_idxs) == 1:
        # we only have one cluster - set the mean and precision
        vapa = edges[cluster_input_idxs[0]].value_parents[0]
        vopa = edges[cluster_input_idxs[0]].volatility_parents[0]

        attributes[vapa]["mean"] = value
        attributes[vapa]["expected_mean"] = value
        attributes[vopa]["mean"] = 1.0
        attributes[vopa]["expected_mean"] = 1.0
    else:
        print(f"Step 2 - {attributes[1]['expected_precision']}")
        # other clusters exist
        attributes = parametrize_cluster_fn(
            value=value,
            input_idx=temp_idx,
            attributes=attributes,
            edges=edges,
            cluster_input_idxs=cluster_input_idxs,
        )
        print(f"Step 3 - {attributes[1]['expected_precision']}")

    # evaluate the likelihood of the current observation under all available branches
    # i.e. pre-existing cluster and temporary one
    clusters_log_likelihood = []
    for input_idx in cluster_input_idxs:
        likelihood = log_likelihood_fn(
            input_idx=input_idx, value=value, attributes=attributes, edges=edges
        )
        clusters_log_likelihood.append(likelihood)

    # probability of being assigned to a pre-existing cluster
    pi_clusters = [
        n / (alpha + attributes[node_idx]["n_total"]) for n in attributes[node_idx]["n"]
    ]

    # probability to draw a new cluster
    pi_new = alpha / (alpha + attributes[node_idx]["n_total"])

    # the probability for a new cluster is attributed to the temporary cluster
    pi_clusters[-1] = pi_new

    # the joint log-likelihoods (evidence + probability)
    clusters_log_likelihood = np.array(clusters_log_likelihood) + np.log(
        np.array(pi_clusters)
    )

    # decide which branch should be updated
    update_idx = np.argmax(clusters_log_likelihood)

    # belief propagation step
    # -----------------------

    # increment the number of observations for the given branch
    attributes[node_idx]["n"][update_idx] += 1

    # mark all branches unobserved
    for input_idx in cluster_input_idxs:
        attributes[input_idx]["observed"] = 0.0

    # if a new cluster was created, create a new temporary one
    if update_idx == attributes[node_idx]["n_clusters"]:
        attributes[node_idx]["n_clusters"] += 1
        attributes, edges, input_nodes_idx, dirichlet_node = create_cluster_fn(
            attributes=attributes,
            edges=edges,
            input_nodes_idx=input_nodes_idx,
            base_network=base_network,  # type: ignore
            dirichlet_node_idx=node_idx,
            dirichlet_node=dirichlet_node,
        )
    else:
        # otherwise, pass the new observation and
        # ensure that the beliefs will propagate in the branch
        update_branch_idx = cluster_input_idxs[int(update_idx)]
        attributes[update_branch_idx]["observed"] = 1.0
        attributes[update_branch_idx]["value"] = value

    attributes[node_idx]["n_total"] += 1

    return attributes, edges, input_nodes_idx, dirichlet_node


def candidate_likelihood(
    sufficient_statistics: Tuple,
    value: float,
    cluster_parameters: Tuple,
    N: int,
    alpha: float,
) -> float:
    """Likelihood of a parametrized candidate under the new observation.

    Parameters
    ----------
    sufficient_statistics :
        The sufficient statistics of the new candidate cluster. These parameter are
        optimized.
    value :
        The value of the new observation.
    cluster_parameters :
        The sufficient statistics and number of observation from the existing clusters.
    N :
        Number of observations in each cluster.
    alpha :
        The concentration parameter of the implied Chinese Restaurant Process.

    Returns
    -------
    neg_likelihood :
        The negative likelihood.

    """
    mu, tau = sufficient_statistics  # mean and log-precision
    scale = np.sqrt(1 / np.exp(tau))

    # likelihood of new observation under the proposal
    cluster_evidence = norm(loc=mu, scale=scale).pdf(value)

    # generate the x vector under which we want to test the distributions
    x = norm(loc=0.0, scale=1).ppf(np.arange(0.001, 1.0, 0.001))

    # new cluster pdf
    cluster_coverage = norm(loc=mu, scale=scale).pdf(x)

    # scale using the expected ratio at t+1 (therefore n=1 for the new cluster)
    cluster_coverage *= 1 / (np.sum(N) + alpha + 1)

    mixture_coverage = []
    ns, means, precisions = cluster_parameters
    for nk, mk, tk in zip(ns, means, precisions):
        mixture_coverage.append(
            (nk / (np.sum(N) + alpha + 1))
            * norm(loc=mk, scale=np.sqrt(1 / (tk))).pdf(x)
        )
    mixture_coverage = np.array(mixture_coverage).sum(axis=0)

    return -np.sum(
        cluster_evidence * cluster_coverage / (cluster_coverage + mixture_coverage)
    )


def parametrize_cluster_fn(
    input_idx: int,
    value: float,
    attributes: Dict,
    edges: Edges,
    cluster_input_idxs: ArrayLike,
) -> Dict:
    """Parametrize a new candidate cluster under the current observation.

    Parameters
    ----------
    input_idx :
        Index of the input node of the cluster to be parametrized.
    value :
        The value of the new observation.
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    cluster_input_idxs :
        The indexes of the existing clusters.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    """
    # retrieve ns, means and precisions from the existing clusters
    means, precisions = [], []
    for i in cluster_input_idxs[:-1]:
        # get mean
        means.append(attributes[edges[i].value_parents[0]]["expected_mean"])

        # get precision
        precisions.append(attributes[i]["expected_precision"])

    ns = attributes[0]["n"]  # number of observation per cluster

    # get cluster candidate
    likelihood_fn = partial(
        candidate_likelihood,
        value=value,
        cluster_parameters=(ns, means, precisions),
        N=attributes[0]["n_total"],
        alpha=attributes[0]["alpha"],
    )
    mu, tau = minimize(likelihood_fn, (value, 1.0))["x"]

    # parametrize a distribution using the candidate parameters
    value_parent_idx = edges[input_idx].value_parents[0]  # type: ignore
    attributes[value_parent_idx]["mean"] = mu
    attributes[value_parent_idx]["expected_mean"] = mu

    mu_volatility = -(np.log(attributes[input_idx]["input_precision"]) + 0.5 * -tau)
    volatility_parent_idx = edges[input_idx].volatility_parents[0]  # type: ignore
    attributes[volatility_parent_idx]["mean"] = mu_volatility
    attributes[volatility_parent_idx]["expected_mean"] = mu_volatility

    return attributes


def create_cluster_fn(
    attributes: Dict,
    edges: Edges,
    input_nodes_idx: InputIndexes,
    base_network: "HGF",
    dirichlet_node: DirichletNode,
    dirichlet_node_idx: int = 0,
) -> Tuple:
    """Create and parametrize a new cluster given the observed values.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.
    base_network :
        An HGF network that will be used as a template to create new branches.
    dirichlet_node :
        Static parameters for the Dirichlet node.
    dirichlet_node_idx :
        Index of the Dirichlet node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the neural network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index lists the value/volatility parents/children.
    input_nodes_idx :
        Static input nodes' parameters for the neural network.
    dirichlet_node:
        Static parameters for the Dirichlet node.

    """
    from pyhgf.networks import add_edges, concatenate_networks

    base_ = copy.deepcopy(base_network)

    network_size = len(attributes)

    # the input indexes for the new branch
    cluster_input_idxs = [idx + network_size for idx in base_.input_nodes_idx.idx]
    new_cluster_input_idxs = dirichlet_node.cluster_input_idxs + tuple(
        cluster_input_idxs
    )
    dirichlet_node = dirichlet_node._replace(cluster_input_idxs=new_cluster_input_idxs)

    # merge the new branch with the existing one
    attributes, edges = concatenate_networks(
        attributes_1=base_.attributes,
        attributes_2=copy.deepcopy(attributes),
        edges_1=base_.edges,
        edges_2=edges,
    )

    # add count for this new cluster
    attributes[dirichlet_node_idx]["n"].append(0)

    # create a new input_nodes variable
    if input_nodes_idx is None:
        input_nodes_idx = base_.input_nodes_idx
    else:
        assert isinstance(input_nodes_idx, InputIndexes)
        new_idx = input_nodes_idx.idx
        new_idx += (base_.input_nodes_idx.idx[0] + network_size,)

        new_kind = input_nodes_idx.kind
        new_kind += base_.input_nodes_idx.kind
        input_nodes_idx = InputIndexes(new_idx, new_kind)

    # add a value coupling between the Dirichlet node and the branch input(s)
    attributes, edges = add_edges(
        attributes=attributes,
        edges=edges,
        kind="value",
        parent_idxs=cluster_input_idxs,
        children_idxs=dirichlet_node_idx,
    )

    return attributes, edges, input_nodes_idx, dirichlet_node

