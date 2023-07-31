# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import TYPE_CHECKING

import jax.numpy as jnp

from pyhgf.continuous import gaussian_surprise

if TYPE_CHECKING:
    from pyhgf.model import HGF


def first_level_gaussian_surprise(hgf: "HGF", response_function_parameters=None):
    """Gaussian surprise at the first level of a probabilistic network.

    .. note::
      The Gaussian surprise at the first level is the default method to compute surprise
      for continuous models. The function returns `jnp.inf` if the model could not fit
      at a given time point.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_parameters :
        No additional parameters are required to compute the Gaussian surprise.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    # compute the sum of Gaussian surprise at the first level
    # the input value at time t is compared to the Gaussian prediction at t-1
    surprise = jnp.sum(
        gaussian_surprise(
            x=hgf.node_trajectories[0]["value"][1:],
            muhat=hgf.node_trajectories[1]["muhat"][:-1],
            pihat=hgf.node_trajectories[1]["pihat"][:-1],
        )
    )

    # Return an infinite surprise if the model could not fit at any point
    return jnp.where(
        jnp.any(jnp.isnan(hgf.node_trajectories[1]["mu"])), jnp.inf, surprise
    )


def total_gaussian_surprise(hgf: "HGF", response_function_parameters=None):
    """Sum of the Gaussian surprise across the probabilistic network.

    .. note::
      The function returns `jnp.inf` if the model could not fit at a given time point.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_parameters :
        No additional parameters are required to compute the Gaussian surprise.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    # compute the sum of Gaussian surprise across every node
    surprise = 0.0

    # first we start with nodes that are value parents to input nodes
    input_parents_list = []
    for idx in hgf.input_nodes_idx.idx:
        va_pa = hgf.node_structure[idx].value_parents[0]  # type: ignore
        input_parents_list.append(va_pa)
        surprise += jnp.sum(
            gaussian_surprise(
                x=hgf.node_trajectories[idx]["value"][1:],
                muhat=hgf.node_trajectories[va_pa]["muhat"][:-1],
                pihat=hgf.node_trajectories[va_pa]["pihat"][:-1],
            )
        )

    # then we do the same for every node that is not an input node
    # and not the parent of an input node
    for i in range(len(hgf.node_structure)):
        if (i not in hgf.input_nodes_idx.idx) and (i not in input_parents_list):
            surprise += jnp.sum(
                gaussian_surprise(
                    x=hgf.node_trajectories[i]["mu"][1:],
                    muhat=hgf.node_trajectories[i]["muhat"][:-1],
                    pihat=hgf.node_trajectories[i]["pihat"][:-1],
                )
            )

    # Return an infinite surprise if the model could not fit at any point
    return jnp.where(
        jnp.any(jnp.isnan(hgf.node_trajectories[1]["mu"])), jnp.inf, surprise
    )


def first_level_binary_surprise(hgf: "HGF", response_function_parameters=None):
    """Sum of the binary surprise along the time series (binary HGF).

    .. note::
      The binary surprise is the default method to compute surprise when
      `model_type=="binary"`, therefore this method will only return the sum of
      valid time points, and `jnp.inf` if the model could not fit.

    Parameters
    ----------
    hgf :
        An instance of the HGF model.
    response_function_parameters :
        No additional parameters are required to compute the binary surprise.

    Returns
    -------
    surprise :
        The model's surprise given the input data.

    """
    # Return an infinite surprise if the model cannot fit
    this_surprise = jnp.where(
        jnp.isnan(hgf.node_trajectories[0]["surprise"]),
        jnp.inf,
        hgf.node_trajectories[0]["surprise"],
    )

    # Sum the surprise for this model
    surprise = jnp.sum(this_surprise)

    return surprise
