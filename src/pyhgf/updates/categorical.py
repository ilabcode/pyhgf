# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import Array, jit
from jax.scipy.special import digamma, gamma
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise
from pyhgf.typing import Edges


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
    # get the expected values at time k from the binary inputs (X_1)
    new_xi = jnp.array(
        [
            attributes[edges[vapa].value_parents[0]]["muhat"]
            for vapa in edges[node_idx].value_parents  # type: ignore
        ]
    )

    # the differential of expectations (parents predictions at time k and k-1)
    delta_xi = new_xi - attributes[node_idx]["xi"]

    # using the PE for the previous time point, we can compute nu and the alpha vector
    pe = attributes[node_idx]["pe"]
    nu = (pe / delta_xi) - 1
    alpha = (nu * new_xi) + 1  # concentration parameters for the Dirichlet

    # in case alpha contains NaNs (e.g. after the first time step,
    # due to the absence of belief evolution)
    alpha = jnp.where(jnp.isnan(alpha), 1.0, alpha)

    # now retrieve the values observed at time k
    attributes[node_idx]["value"] = jnp.array(
        [
            attributes[vapa]["value"]
            for vapa in edges[node_idx].value_parents  # type: ignore
        ]
    )

    # compute the prediction error at time K
    pe = attributes[node_idx]["value"] - new_xi
    attributes[node_idx]["pe"] = pe  # keep PE for later use at k+1
    attributes[node_idx]["xi"] = new_xi  # keep expectation for later use at k+1

    # compute Bayesian surprise as :
    # 1 - KL divergence from the concentration parameters
    # 2 - the sum of binary surprises observed in the parents nodes
    attributes[node_idx]["kl_divergence"] = dirichlet_kullback_leibler(
        attributes[node_idx]["alpha"], alpha
    )
    attributes[node_idx]["binary_surprise"] = jnp.sum(
        binary_surprise(
            x=attributes[node_idx]["value"], muhat=attributes[node_idx]["xi"]
        )
    )

    # save the new concentration parameters
    attributes[node_idx]["alpha"] = alpha

    # prediction mean
    attributes[node_idx]["mu"] = alpha / jnp.sum(alpha)
    attributes[node_idx]["time_step"] = time_step

    return attributes


def dirichlet_kullback_leibler(alpha_1: ArrayLike, alpha_2: ArrayLike) -> Array:
    r"""Compute the Kullback-Leibler divergence between two Dirichlet distributions.

    The Kullback-Leibler divergence from the distribution :math:`Q` to the distribution
    :math:`P`, two Dirichlet distributions parametrized by :math:`\\alpha_2` and
    :math:`\\alpha_1` (respectively) is given by the following equation:

    .. math::
       KL[P||Q] = \\
       \\ln{\\frac{\\Gamma(\\sum_{i=1}^k\\alpha_{1i})} \\
       {\\Gamma(\\sum_{i=1}^k\\alpha_{2i})}} + \\
       \\sum_{i=1}^k \\ln{\\frac{\\Gamma(\\alpha_{2i})}{\\Gamma(\\alpha_{1i})}} + \\
       \\sum_{i=1}^k(\\alpha_{1i} -\\
       \\alpha_{2i})\\left[\\psi(\alpha_{1i})-\\psi(\\sum_{i=1}^k\alpha_{1i})\\right]

    Parameters
    ----------
    alpha_1 :
        The concentration parameters for the distribution :math:`P`.
    alpha_2 :
        The concentration parameters for the distribution :math:`Q`.

    Returns
    -------
    kl :
        The Kullback-Leibler divergence of distribution :math:`P` from distribution
        :math:`Q`.

    References
    ----------
    .. [1] https://statproofbook.github.io/P/dir-kl.html
    .. [2] Penny, William D. (2001): "KL-Divergences of Normal, Gamma, Dirichlet and
       Wishart densities" ; in: University College, London , p. 2, eqs. 8-9 ;
       URL: https://www.fil.ion.ucl.ac.uk/~wpenny/publications/densities.ps .

    """
    return (
        jnp.log(gamma(alpha_1.sum()) / gamma(alpha_2.sum()))
        + jnp.sum(jnp.log(gamma(alpha_2) / gamma(alpha_1)))
        + jnp.sum((alpha_1 - alpha_2) * (digamma(alpha_1) - digamma(alpha_1.sum())))
    )
