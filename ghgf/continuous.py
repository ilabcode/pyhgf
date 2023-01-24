# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Any, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.interpreters.xla import DeviceArray

from ghgf.typing import NodeType, ParametersType, ParentsType


def continuous_node_update(
    node_parameters: ParametersType,
    value_parents: ParentsType,
    volatility_parents: ParentsType,
    old_time: Union[float, DeviceArray],
    new_time: Union[float, DeviceArray],
) -> NodeType:
    """Update the value and volatility parents of a continuous node.

    If the parents have value and/or volatility parents, they will be updated
    recursively.

    Updating the node's parents is a two step process:
        1. Update value parent(s) and their parents (if provided).
        2. Update volatility parent(s) and their parents (if provided).

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

    Parameters
    ----------
    node_parameters : dict
        The node parameters is a dictionary containing the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.

        `"psis"` is a tuple with same length than value parents.
        `"kappas"` is a tuple with same length than volatility parents.
    value_parents : tuple | None
        The value parent(s) (optional).
    volatility_parents : tuple | None
        The volatility parent(s) (optional).
    old_time : DeviceArray
        The previous time stamp.
    new_time : DeviceArray
        The previous time stamp.

    Returns
    -------
    node_parameters : dict
        The new node parameters containing the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.
    value_parents : tuple | None
        The new value parent (optional).
    volatility_parents : tuple | None
        The new volatility parent (optional).
    old_time : DeviceArray
        The previous time stamp.
    new_time : DeviceArray
        The previous time stamp.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    """
    # Return here if no parents node are provided
    if (value_parents is None) and (volatility_parents is None):
        return node_parameters, value_parents, volatility_parents

    # Time interval
    t = jnp.subtract(new_time, old_time)

    pihat: DeviceArray = node_parameters["pihat"]

    #######################################
    # Update the continuous value parents #
    #######################################
    if value_parents is not None:

        vape = node_parameters["mu"] - node_parameters["muhat"]
        psis = node_parameters["psis"]

        new_value_parents: Any = ()
        for va_pa, psi in zip(value_parents, psis):  # type:ignore

            # Unpack this node's parameters, value and volatility parents
            va_pa_node_parameters, va_pa_value_parents, va_pa_volatility_parents = va_pa

            # Compute new value for nu and pihat
            logvol = va_pa_node_parameters["omega"]

            # Look at the (optional) va_pa's volatility parents
            # and update logvol accordingly
            if va_pa_volatility_parents is not None:
                for va_pa_vo_pa, k in zip(
                    va_pa_volatility_parents, va_pa_node_parameters["kappas"]
                ):
                    logvol += k * va_pa_vo_pa[0]["mu"]

            # Estimate new_nu
            nu = t * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_pa, nu_pa = [1 / (1 / va_pa_node_parameters["pi"] + new_nu), new_nu]
            pi_pa = pihat_pa + psi**2 * pihat

            # Compute new muhat
            driftrate = va_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            if va_pa_value_parents is not None:
                for va_pa_va_pa in va_pa_value_parents:
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
            ) = continuous_node_update(
                node_parameters=va_pa_node_parameters,
                value_parents=va_pa_value_parents,
                volatility_parents=va_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_value_parents += (
                (
                    new_va_pa_node_parameters,
                    new_va_pa_value_parents,
                    new_va_pa_volatility_parents,
                ),
            )

    else:
        new_value_parents = value_parents

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents is not None:

        nu = node_parameters["nu"]
        kappas = node_parameters["kappas"]
        vope = (
            1 / node_parameters["pi"]
            + (node_parameters["mu"] - node_parameters["muhat"]) ** 2
        ) * node_parameters["pihat"] - 1

        new_volatility_parents = ()  # type:ignore
        for vo_pa, kappa in zip(volatility_parents, kappas):  # type:ignore

            # Unpack this node's parameters, value and volatility parents
            vo_pa_node_parameters, vo_pa_value_parents, vo_pa_volatility_parents = vo_pa

            # Compute new value for nu and pihat
            logvol = vo_pa_node_parameters["omega"]

            # Look at the (optional) vo_pa's volatility parents
            # and update logvol accordingly
            if vo_pa_volatility_parents is not None:
                for i, vo_pa_vo_pa in enumerate(vo_pa_volatility_parents):
                    k = vo_pa_node_parameters["kappas"][i]
                    logvol += k * vo_pa_vo_pa[0]["mu"]

            # Estimate new_nu
            new_nu = t * jnp.exp(logvol)
            new_nu = jnp.where(new_nu > 1e-128, new_nu, jnp.nan)

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
            if vo_pa_value_parents is not None:
                for vo_pa_va_pa in vo_pa_value_parents:
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
            ) = continuous_node_update(
                node_parameters=vo_pa_node_parameters,
                value_parents=vo_pa_value_parents,
                volatility_parents=vo_pa_volatility_parents,
                old_time=old_time,
                new_time=new_time,
            )

            new_volatility_parents += (  # type:ignore
                (
                    new_vo_pa_node_parameters,
                    new_vo_pa_value_parents,
                    new_vo_pa_volatility_parents,
                ),
            )
    else:
        new_volatility_parents = volatility_parents  # type:ignore

    ##########################
    # Update node parameters #
    ##########################
    new_node_parameters = node_parameters

    return new_node_parameters, new_value_parents, new_volatility_parents


def continuous_input_update(
    input_node: NodeType,
    value: DeviceArray,
    new_time: DeviceArray,
    old_time: DeviceArray,
) -> Optional[Tuple[DeviceArray, NodeType]]:
    """Update the input node structure.

    This function is the entry level of the model fitting. It update the partents of
    the input node and then call py:func:`ghgf.jax.continuous_node_update` recursively
    to update the rest of the node structure.

    Parameters
    ----------
    input_node : tuple
        The parameter, value parent and volatility parent of the input node. The
        volatility and value parents contain their own value and volatility parents,
        this structure being nested up to the higher level of the hierarchy.
    value : DeviceArray
        The new input value that is observed by the model at time t=`new_time`.
    new_time : DeviceArray
        The current time point.
    old_time : DeviceArray
        The time point of the previous observed value.

    Returns
    -------
    surprise : jnp.DeviceArray
        The gaussian surprise given the value(s) presented at t=`new_time`.
    new_input_node : tuple
        The input node structure after recursively updating all the nodes.

    See Also
    --------
    continuous_node_update, update_binary_input_parents

    """
    input_node_parameters, value_parents, volatility_parents = input_node

    if (value_parents is None) and (volatility_parents is None):
        return None

    # Time interval
    t = jnp.subtract(new_time, old_time)

    # Add a bias to the input value
    if input_node_parameters["bias"] is not None:
        value = jnp.add(value, input_node_parameters["bias"])

    lognoise = input_node_parameters["omega"]

    if input_node_parameters["kappas"] is not None:
        lognoise += (  # type:ignore
            input_node_parameters["kappas"]
            * volatility_parents[0][0]["mu"]  # type:ignore
        )

    pihat = 1 / jnp.exp(lognoise)

    ########################
    # Update value parents #
    ########################
    if value_parents is not None:

        # Unpack this node's parameters, value and volatility parents
        (
            va_pa_node_parameters,
            va_pa_value_parents,
            va_pa_volatility_parents,
        ) = value_parents[0]

        vape = va_pa_node_parameters["mu"] - va_pa_node_parameters["muhat"]

        # Compute new value for nu and pihat
        logvol = va_pa_node_parameters["omega"]

        # Look at the (optional) va_pa's volatility parents
        # and update logvol accordingly
        if va_pa_volatility_parents is not None:
            for va_pa_vo_pa, k in zip(
                va_pa_volatility_parents, va_pa_node_parameters["kappas"]
            ):
                logvol += k * va_pa_vo_pa[0]["mu"]

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
        if va_pa_value_parents is not None:
            for va_pa_va_pa in va_pa_value_parents:
                driftrate += va_pa_node_parameters["psi"] * va_pa_va_pa[0]["mu"]

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
        ) = continuous_node_update(
            node_parameters=va_pa_node_parameters,
            value_parents=va_pa_value_parents,
            volatility_parents=va_pa_volatility_parents,
            old_time=old_time,
            new_time=new_time,
        )

        new_value_parents = (
            (
                new_va_pa_node_parameters,
                new_va_pa_value_parents,
                new_va_pa_volatility_parents,
            ),
        )

    else:
        new_value_parents = value_parents  # type:ignore

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents is not None:

        # Unpack this node's parameters, value and volatility parents
        (
            vo_pa_node_parameters,
            vo_pa_value_parents,
            vo_pa_volatility_parents,
        ) = volatility_parents[0]

        vope = (1 / pi_va_pa + (value - mu_va_pa) ** 2) * pihat - 1

        nu = vo_pa_node_parameters["nu"]

        # Compute new value for nu and pihat
        logvol = vo_pa_node_parameters["omega"]

        # Look at the (optional) vo_pa's volatility parents
        # and update logvol accordingly
        if vo_pa_volatility_parents is not None:
            for vo_pa_vo_pa in vo_pa_volatility_parents:
                kappa = vo_pa_vo_pa[0]["kappas"]
                logvol += kappa * vo_pa_vo_pa[0]["mu"]

        # Estimate new_nu
        nu = t * jnp.exp(logvol)
        new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

        pihat_vo_pa, nu_vo_pa = [
            1 / (1 / vo_pa_node_parameters["pi"] + new_nu),
            new_nu,
        ]

        pi_vo_pa = pihat_vo_pa + 0.5 * jnp.square(input_node_parameters["kappas"]) * (
            1 + vope
        )
        pi_vo_pa = jnp.where(pi_vo_pa <= 0, jnp.nan, pi_vo_pa)

        # Compute new muhat
        driftrate = vo_pa_node_parameters["rho"]

        # Look at the (optional) va_pa's value parents
        # and update driftrate accordingly
        if vo_pa_value_parents is not None:
            for vo_pa_va_pa, p in zip(
                vo_pa_value_parents, vo_pa_node_parameters["psis"]
            ):
                driftrate += p * vo_pa_va_pa[0]["mu"]

        muhat_vo_pa = vo_pa_node_parameters["mu"] + t * driftrate
        mu_vo_pa = (
            muhat_vo_pa
            + jnp.multiply(0.5, input_node_parameters["kappas"]) / pi_vo_pa * vope
        )

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
        ) = continuous_node_update(
            node_parameters=vo_pa_node_parameters,
            value_parents=vo_pa_value_parents,
            volatility_parents=vo_pa_volatility_parents,
            old_time=old_time,
            new_time=new_time,
        )

        new_volatility_parents = (
            (
                new_vo_pa_node_parameters,
                new_vo_pa_value_parents,
                new_vo_pa_volatility_parents,
            ),
        )

    else:
        new_volatility_parents = volatility_parents  # type:ignore

    surprise = gaussian_surprise(x=value, muhat=muhat_va_pa, pihat=pihat)
    input_node = input_node_parameters, new_value_parents, new_volatility_parents

    return surprise, input_node


def gaussian_surprise(
    x: jnp.DeviceArray, muhat: jnp.DeviceArray, pihat: jnp.DeviceArray
) -> jnp.DeviceArray:
    """Surprise at an outcome under a Gaussian prediction."""
    return jnp.array(0.5) * (
        jnp.log(jnp.array(2.0) * jnp.pi)
        - jnp.log(pihat)
        + pihat * jnp.square(jnp.subtract(x, muhat))
    )


@jit
def loop_continuous_inputs(res: Tuple, el: Tuple) -> Tuple[Tuple, Tuple]:
    """Update the node structure after observing one new data point.

    One time step updating node structure and returning the new node structure with
    time, value and surprise.

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
    surprise, new_node_structure = continuous_input_update(  # type:ignore
        input_node=node_structure,
        value=value,
        old_time=results["time"],
        new_time=new_time,
    )

    # the model structure is packed with current time point
    # surprise is accumulated separately
    res = new_node_structure, {"time": new_time, "value": value, "surprise": surprise}

    return res, res  # ("carryover", "accumulated")
