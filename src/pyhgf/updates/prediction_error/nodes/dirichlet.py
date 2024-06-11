# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import Array, jit, random
from jax._src.typing import Array as KeyArray
from jax.scipy.stats.norm import pdf
from jax.typing import ArrayLike

from pyhgf.typing import Attributes, Edges


def dirichlet_node_prediction_error(
    edges: Edges,
    attributes: Dict,
    value: float,
    node_idx: int,
    **args,
) -> Attributes:
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
    values = attributes[node_idx]["values"]  # the input value
    alpha = attributes[node_idx]["alpha"]  # the concentration parameter
    n_total = attributes[node_idx]["n_total"]  # total number of observations
    n = attributes[node_idx]["n"]  # number of observations per cluster
    sensory_precision = attributes[node_idx][
        "sensory_precision"
    ]  # number of observations per cluster

    # likelihood of the current observation under existing clusters
    # -------------------------------------------------------------
    cluster_ll = clusters_likelihood(
        value=values,
        expected_mean=attributes[node_idx]["expected_mean"],
        expected_sigma=attributes[node_idx]["expected_sigma"],
    )

    # set the likelihood to 0 for inactive clusters
    cluster_ll *= attributes[node_idx]["activated"]

    # likelihood of the current observation under the best candidate cluster
    # ----------------------------------------------------------------------

    # find the best cluster candidate given the new observation
    candidate_mean, candidate_sigma = get_candidate(
        value=value,
        sensory_precision=sensory_precision,
        expected_mean=attributes[node_idx]["expected_mean"],
        expected_sigma=attributes[node_idx]["expected_sigma"],
    )

    # get the likelihood under this candidate
    candidate_ll = clusters_likelihood(
        value=value,
        expected_mean=candidate_mean,
        expected_sigma=candidate_sigma,
    )

    # DP step: compare the likelihood of existing cluster with a new cluster
    # ----------------------------------------------------------------------

    # probability of being assigned to a pre-existing cluster
    cluster_ll *= n / (alpha + n_total)

    # probability to draw a new cluster
    candidate_ll *= alpha / (alpha + n_total)

    best_idx, best_val = jnp.argmax(cluster_ll), jnp.max(cluster_ll)

    # set all cluster to non-observed by default
    for parent_idx in edges[node_idx].value_parents:  # type:ignore
        attributes[parent_idx]["observed"] = 0.0

    if best_val >= candidate_ll:

        # activate the corresponding branch and pass the value
        active_idx = edges[node_idx].value_parents[best_idx]  # type:ignore
        attributes[active_idx]["observed"] = 1.0
        attributes[active_idx]["values"] = value

    else:

        # activate a new cluster
        attributes[node_idx]["n_active_cluster"] += 1
        active_idx = attributes[node_idx]["n_active_cluster"]
        attributes[node_idx]["activated"].at[active_idx].set(1.0)
        attributes[active_idx]["observed"] = 1.0
        attributes[active_idx]["values"] = value

    return attributes


@jit
def get_candidate(
    value: float,
    sensory_precision: float,
    expected_mean: ArrayLike,
    expected_sigma: ArrayLike,
    n_samples: int = 20_000,
) -> Tuple[float, float]:
    """Find the best cluster candidate given previous clusters and an input value.

    Parameters
    ----------
    value :
        The new observation.
    sensory_precision :
        The expected precision of the new observation.
    expected_mean :
        The mean of the existing clusters.
    expected_sigma :
        The standard deviation of the existing clusters.
    n_samples :
        The number of samples that should be simulated.

    Returns
    -------
    mean :
        The mean of the new candidate cluster.
    sigma :
        The standard deviation of the new candidate cluster.

    """
    # initialize weigths
    weigths = jnp.ones(n_samples)

    # sample n likely clusters given the base distribution priors
    mus, sigmas, weigths = likely_cluster_proposal(
        mean_mu_G0=0.0,
        sigma_mu_G0=100.0,
        sigma_pi_G0=5.0,
        expected_mean=expected_mean,
        expected_sigma=expected_sigma,
        key=random.key(42),
        n_samples=n_samples,
    )

    # 1 - Likelihood of the new observation under each sampled cluster
    # ----------------------------------------------------------------
    ll_value = pdf(value, mus, sigmas)
    ll_value /= ll_value.sum()  # normalize the weigths

    # 2- re-scale the weigths using expected precision
    # ------------------------------------------------
    weigths *= ll_value**sensory_precision

    # 3 - estimate new mean and standard deviation using the weigthed mean
    # --------------------------------------------------------------------
    mean = ((mus * weigths) / weigths.sum()).sum()
    sigma = ((sigmas * weigths) / weigths.sum()).sum()

    return mean, sigma


@partial(jit, static_argnames=("n_samples"))
def likely_cluster_proposal(
    mean_mu_G0: float,
    sigma_mu_G0: float,
    sigma_pi_G0: float,
    expected_mean=ArrayLike,
    expected_sigma=ArrayLike,
    key: KeyArray = random.key(42),
    n_samples: int = 20_000,
) -> Tuple[Array, Array, Array]:
    """Sample likely new belief distributions given pre-existing clusters.

    Parameters
    ----------
    mean_mu_G0 :
        The mean of the mean of the base distribution.
    sigma_mu_G0 :
        The standard deviation of mean of the base distribution.
    sigma_pi_G0 :
        The standard deviation of the precision of the base distribution.
    expected_mean :
        Pre-existing clusters means.
    expected_sigma :
        Pre-existing clusters standard deviation.
    key :
        Random state.
    n_samples :
        The number of samples used during the simulations.

    Returns
    -------
    new_mu :
        A vector of means candidates.
    new_sigma :
        A vector of standard deviation candidates.
    weigths :
        Weigths for each cluster candidate under pre-existing cluster (irrespective of
        new observations).

    """
    # sample new candidate for cluster means
    key, use_key = random.split(key)
    new_mu = sigma_mu_G0 * random.normal(use_key, (n_samples,)) + mean_mu_G0

    # sample new candidate for cluster standard deviation
    key, use_key = random.split(key)
    new_sigma = jnp.abs(random.normal(use_key, (n_samples,)) * sigma_pi_G0)

    # initialize the weigths
    weigths = jnp.zeros(n_samples)

    # 1 - Cluster specificity
    # -----------------------
    # this cluster should explain new dimensions, not explained by other clusters

    # evidence for pre-existing clusters
    pre_existing_likelihood = jnp.zeros(n_samples)
    for mu_i, sigma_i in zip(expected_mean, expected_sigma):
        pre_existing_likelihood += pdf(new_mu, mu_i, sigma_i)

    # evidence for the new cluster proposal
    new_likelihood = pdf(new_mu, new_mu, new_sigma)

    # standardize the measure of cluster specificity (ratio)
    ratio = new_likelihood / (new_likelihood + pre_existing_likelihood)
    ratio = jnp.where(ratio <= 0.0, 0.0, ratio)

    weigths += ratio

    # 2 - Cluster isolation
    # ---------------------
    # this cluster should not try to explain what was already explained

    # (pre-existing cluster) / (pre-existing cluster + new cluster)
    cluster_isolation = jnp.ones(n_samples)
    for mu_i, sigma_i in zip(expected_mean, expected_sigma):
        ratio = pdf(mu_i, mu_i, sigma_i) / (
            pdf(mu_i, mu_i, sigma_i) + pdf(mu_i, new_mu, new_sigma)
        )
        ratio = jnp.where(ratio <= 0.0, 0.0, ratio)
        cluster_isolation *= ratio

    weigths += cluster_isolation

    return new_mu, new_sigma, weigths


def clusters_likelihood(
    value: float,
    expected_mean: ArrayLike,
    expected_sigma: ArrayLike,
) -> ArrayLike:
    """Likelihood of a parametrized candidate under the new observation.

    Parameters
    ----------
    value :
        The new observation.
    expected_mean :
        Pre-existing clusters means.
    expected_sigma :
        Pre-existing clusters standard deviation.

    Returns
    -------
    likelihood :
        The probability of observing the value under each cluster.

    """
    return pdf(value, expected_mean, expected_sigma)
