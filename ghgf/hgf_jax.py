from jax.interpreters.xla import DeviceArray
import jax.numpy as jnp
from numpy import loadtxt
from jax.lax import scan
from jax import jit
import matplotlib.pyplot as plt
from functools import partial
import os

# from ghgf.hgf import StandardHGF
from typing import Tuple, Union, Optional, List, Dict

# path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
data = loadtxt(f"/home/nicolas/git/ghgf/tests/data/usdchf.dat")


def update_parents(
    node_parameters: Dict[str, float],
    value_parents: List[Optional[List]],
    volatility_parents: List[Optional[List]],
    old_time: int,
    new_time: int,
) -> Tuple[Dict[str, float], Optional[List], Optional[List], int, int]:
    """Update the value parents from a given node. If the node has value or volatility
    parents, they will be updated recursively.

    Parameters
    ----------
    node_parameters : dict
        The node parameters containing the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.

        `"psis"` is a list with same length than value parents.

        `"kappas"` is a list with same length than volatility parents.
    value_parents : list | None
        The value parent (optional).
    volatility_parents : list | None
        The volatility parent (optional).
    old_time : int
        The previous time stamp.
    new_time : int
        The previous time stamp.

    Returns
    -------
    node_parameters : dict
        The new node parameters containing the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.
    value_parents : list | None
        The new value parent (optional).
    volatility_parents : list | None
        The new volatility parent (optional).
    old_time : int
        The previous time stamp.
    new_time : int
        The previous time stamp.

    """
    # Return here if no parents node are provided
    if (value_parents[0] is None) and (volatility_parents[0] is None):
        return node_parameters, value_parents, volatility_parents

    # Time interval
    t = new_time - old_time

    pihat = node_parameters["pihat"]

    ########################
    # Update value parents #
    ########################
    if value_parents[0] is not None:
        vape = node_parameters["mu"] - node_parameters["muhat"]
        psis = node_parameters["psis"]

        new_value_parents = []
        for va_pa, psi in zip(value_parents, psis):

            # Unpack this node's parameters, value and volatility parents
            va_pa_node_parameters, va_pa_value_parents, va_pa_volatility_parents = va_pa

            # Compute new value for nu and pihat
            logvol = va_pa_node_parameters["omega"]

            # Look at the (optional) va_pa's volatility parents
            # and update logvol accordingly
            for va_pa_vo_pa in va_pa_volatility_parents:
                if va_pa_vo_pa is not None:
                    kappa = va_pa_vo_pa[0]["kappas"]
                    logvol += kappa * va_pa_vo_pa["mu"]

            # Estimate new_nu
            nu = t * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_pa, nu_pa = [1 / (1 / va_pa_node_parameters["pi"] + new_nu), new_nu]
            pi_pa = pihat_pa + psi ** 2 * pihat

            # Compute new muhat
            driftrate = va_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            for va_pa_va_pa in va_pa_value_parents:
                if va_pa_va_pa is not None:
                    driftrate += psi * va_pa_va_pa["mu"]

            muhat_pa = va_pa_node_parameters["mu"] + t * driftrate
            mu_pa = muhat_pa + psi * pihat / pi_pa * vape

            # Update node's parameters
            va_pa_node_parameters["pihat"] = pihat_pa
            va_pa_node_parameters["pi"] = pi_pa
            va_pa_node_parameters["muhat"] = muhat_pa
            va_pa_node_parameters["mu"] = mu_pa
            va_pa_node_parameters["nu"] = nu_pa

            # Update node's parents recursively
            (
                new_va_pa_node_parameters,
                new_va_pa_value_parents,
                new_va_pa_volatility_parents,
            ) = update_parents(
                node_parameters=va_pa_node_parameters,
                value_parents=va_pa_value_parents,
                volatility_parents=va_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_value_parents.append(
                [
                    new_va_pa_node_parameters,
                    new_va_pa_value_parents,
                    new_va_pa_volatility_parents,
                ]
            )
    else:
        new_value_parents = value_parents

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents[0] is not None:

        nu = node_parameters["nu"]
        kappas = node_parameters["kappas"]
        vope = (
            1 / node_parameters["pi"]
            + (node_parameters["mu"] - node_parameters["muhat"]) ** 2
        ) * node_parameters["pihat"] - 1

        new_volatility_parents = []
        for vo_pa, kappa in zip(volatility_parents, kappas):

            # Unpack this node's parameters, value and volatility parents
            vo_pa_node_parameters, vo_pa_value_parents, vo_pa_volatility_parents = vo_pa

            # Compute new value for nu and pihat
            logvol = vo_pa_node_parameters["omega"]

            # Look at the (optional) vo_pa's volatility parents
            # and update logvol accordingly
            for vo_pa_vo_pa in vo_pa_volatility_parents:
                if vo_pa_vo_pa is not None:
                    kappa = vo_pa_vo_pa[0]["kappas"]
                    logvol += kappa * vo_pa_vo_pa["mu"]

            # Estimate new_nu
            nu = t * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_pa, nu_pa = [1 / (1 / vo_pa_node_parameters["pi"] + new_nu), new_nu]

            pi_pa = pihat_pa + 0.5 * (kappa * nu * pihat) ** 2 * (
                1
                + (1 - 1 / (nu * node_parameters["pi"]))
                * vope  # !! pi should go 2x back in time? !!
            )
            pi_pa = jnp.where(pi_pa <= 0, jnp.nan, pi_pa)

            # Compute new muhat
            driftrate = vo_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            for vo_pa_va_pa in vo_pa_value_parents:
                if vo_pa_va_pa is not None:
                    driftrate += psi * vo_pa_va_pa["mu"]

            muhat_pa = vo_pa_node_parameters["mu"] + t * driftrate
            mu_pa = muhat_pa + 0.5 * kappa * nu * pihat / pi_pa * vope

            # Update node's parameters
            vo_pa_node_parameters["pihat"] = pihat_pa
            vo_pa_node_parameters["pi"] = pi_pa
            vo_pa_node_parameters["muhat"] = muhat_pa
            vo_pa_node_parameters["mu"] = mu_pa
            vo_pa_node_parameters["nu"] = nu_pa

            # Update node's parents recursively
            (
                new_vo_pa_node_parameters,
                new_vo_pa_value_parents,
                new_vo_pa_volatility_parents,
            ) = update_parents(
                node_parameters=vo_pa_node_parameters,
                value_parents=vo_pa_value_parents,
                volatility_parents=vo_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_volatility_parents.append(
                [
                    new_vo_pa_node_parameters,
                    new_vo_pa_value_parents,
                    new_vo_pa_volatility_parents,
                ]
            )
    else:
        new_volatility_parents = volatility_parents
    ##########################
    # Update node parameters #
    ##########################
    new_node_parameters = node_parameters

    return new_node_parameters, new_value_parents, new_volatility_parents


def update_input_parents(
    input_node: List,
    value: jnp.DeviceArray,
    new_time: jnp.DeviceArray,
    old_time: jnp.DeviceArray,
) -> Union[jnp.DeviceArray, List]:
    """Update node structure given one value and return surprise.
    
    Parameters
    ----------
    `"mu", "muhat", "psis", "omega"`

    Returns
    -------
    surprise : jnp.DeviceArray
        The surprise given the value presented.
    new_input_node : list
        The updated node structure.

    """
    input_node_parameters, value_parents, volatility_parents = input_node

    if (value_parents[0] is None) and (volatility_parents[0] is None):
        return

    # Time interval
    t = new_time - old_time

    lognoise = input_node_parameters["omega"]
    kappa = input_node_parameters["kappas"][0]

    if kappa is not None:
        lognoise += kappa * volatility_parents[0][0]["mu"]

    pihat = 1 / jnp.exp(lognoise)

    ########################
    # Update value parents #
    ########################
    if value_parents[0] is not None:
        vape = input_node_parameters["mu"] - input_node_parameters["muhat"]
        psis = input_node_parameters["psis"]

        new_value_parents = []
        for va_pa, psi in zip(value_parents, psis):

            # Unpack this node's parameters, value and volatility parents
            va_pa_node_parameters, va_pa_value_parents, va_pa_volatility_parents = va_pa

            # Compute new value for nu and pihat
            logvol = va_pa_node_parameters["omega"]

            # Look at the (optional) va_pa's volatility parents
            # and update logvol accordingly
            for va_pa_vo_pa in va_pa_volatility_parents:
                if va_pa_vo_pa is not None:
                    kappa = va_pa_vo_pa[0]["kappas"]
                    logvol += kappa * va_pa_vo_pa["mu"]

            # Estimate new_nu
            nu = t * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_va_pa, nu_va_pa = [
                1 / (1 / va_pa_node_parameters["pi"] + new_nu),
                new_nu,
            ]
            pi_va_pa = pihat_va_pa + pihat

            # Compute new muhat
            driftrate = va_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            for va_pa_va_pa in va_pa_value_parents:
                if va_pa_va_pa is not None:
                    driftrate += psi * va_pa_va_pa["mu"]

            muhat_va_pa = va_pa_node_parameters["mu"] + t * driftrate
            vape = value - muhat_va_pa
            mu_va_pa = muhat_va_pa + pihat / pi_va_pa * vape

            # Update value parent's parameters
            va_pa_node_parameters["pihat"] = pihat_va_pa
            va_pa_node_parameters["pi"] = pi_va_pa
            va_pa_node_parameters["muhat"] = muhat_va_pa
            va_pa_node_parameters["mu"] = mu_va_pa
            va_pa_node_parameters["nu"] = nu_va_pa

            # Update node's parents recursively
            (
                new_va_pa_node_parameters,
                new_va_pa_value_parents,
                new_va_pa_volatility_parents,
            ) = update_parents(
                node_parameters=va_pa_node_parameters,
                value_parents=va_pa_value_parents,
                volatility_parents=va_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_value_parents.append(
                [
                    new_va_pa_node_parameters,
                    new_va_pa_value_parents,
                    new_va_pa_volatility_parents,
                ]
            )
    else:
        new_value_parents = value_parents

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents[0] is not None:

        vope = (1 / pi_va_pa + (value - mu_va_pa) ** 2) * pihat - 1

        nu = input_node_parameters["nu"]
        kappas = input_node_parameters["kappas"]

        new_volatility_parents = []
        for vo_pa, kappa in zip(volatility_parents, kappas):

            # Unpack this node's parameters, value and volatility parents
            vo_pa_node_parameters, vo_pa_value_parents, vo_pa_volatility_parents = vo_pa

            # Compute new value for nu and pihat
            logvol = vo_pa_node_parameters["omega"]

            # Look at the (optional) vo_pa's volatility parents
            # and update logvol accordingly
            for vo_pa_vo_pa in vo_pa_volatility_parents:
                if vo_pa_vo_pa is not None:
                    kappa = vo_pa_vo_pa[0]["kappas"]
                    logvol += kappa * vo_pa_vo_pa["mu"]

            # Estimate new_nu
            nu = t * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_vo_pa, nu_vo_pa = [
                1 / (1 / vo_pa_node_parameters["pi"] + new_nu),
                new_nu,
            ]

            pi_vo_pa = pihat_vo_pa + 0.5 * kappa ** 2 * (1 + vope)
            pi_vo_pa = jnp.where(pi_vo_pa <= 0, jnp.nan, pi_vo_pa)

            # Compute new muhat
            driftrate = vo_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            for vo_pa_va_pa in vo_pa_value_parents:
                if vo_pa_va_pa is not None:
                    driftrate += psi * vo_pa_va_pa["mu"]

            muhat_vo_pa = vo_pa_node_parameters["mu"] + t * driftrate
            mu_vo_pa = muhat_vo_pa + 0.5 * kappa / pi_vo_pa * vope

            # Update node's parameters
            vo_pa_node_parameters["pihat"] = pihat_vo_pa
            vo_pa_node_parameters["pi"] = pi_vo_pa
            vo_pa_node_parameters["muhat"] = muhat_vo_pa
            vo_pa_node_parameters["mu"] = mu_vo_pa
            vo_pa_node_parameters["nu"] = nu_vo_pa

            # Update node's parents recursively
            (
                new_vo_pa_node_parameters,
                new_vo_pa_value_parents,
                new_vo_pa_volatility_parents,
            ) = update_parents(
                node_parameters=vo_pa_node_parameters,
                value_parents=vo_pa_value_parents,
                volatility_parents=vo_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_volatility_parents.append(
                [
                    new_vo_pa_node_parameters,
                    new_vo_pa_value_parents,
                    new_vo_pa_volatility_parents,
                ]
            )
    else:
        new_volatility_parents = volatility_parents

    surprise = gaussian_surprise(x=value, muhat=muhat_va_pa, pihat=pihat)
    input_node = [input_node_parameters, new_value_parents, new_volatility_parents]

    return surprise, input_node


def gaussian_surprise(
    x: jnp.DeviceArray, muhat: jnp.DeviceArray, pihat: jnp.DeviceArray
) -> jnp.DeviceArray:
    """Surprise at an outcome under a Gaussian prediction.
    
    Parameters
    ----------
    x : jnp.DeviceArray
        The outcome.
    muhat : jnp.DeviceArray
        The mean of the Gaussian distribution.
    pihat : jnp.DeviceArray
        The precision of the Gaussian distribution.

    Return
    ------
    surprise : jnp.DeviceArray
        The surprise.

    """
    return jnp.array(0.5) * (
        jnp.log(jnp.array(2.0) * jnp.pi)
        - jnp.log(pihat)
        + pihat * jnp.square(x - muhat)
    )


def loop_inputs(
    res: Tuple, el: Tuple
) -> Tuple[Tuple[List, Dict[str, DeviceArray]], DeviceArray]:
    """The HGF function to be scanned by JAX. One time step updating node structure and
    returning time, value and surprise.

    Parameters
    ----------
    res : tuple
        Result from the previous iteration. The node structure and results from the
        previous iteration as a tuple in the form:
        `(node_structure, {"time": time, "value": value, "surprise": surprise})`.
    el : tuple
        The current array element. Scan will iterate over a n x 2 DeviceArray with time
        and values (i.e. the input time series).

    """

    # Extract the current iteration variables values and time
    value, new_time = el

    # Extract model previous structure and results {time, value, surprise}
    node_structure, results = res

    # Fit the model with the current time and value variables, given model structure
    surprise, new_node_structure = update_input_parents(
        input_node=node_structure,
        value=value,
        old_time=results["time"],
        new_time=new_time,
    )

    # the model structure is packed with current time point
    # surprise is accumulated separately
    res = new_node_structure, {"time": new_time, "value": value, "surprise": surprise}

    return res, res  # ("carryover", "accumulated")
