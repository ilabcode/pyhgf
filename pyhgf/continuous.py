# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnames=("node_structure", "node_idx"))
def continuous_node_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: Tuple,
    **args
) -> Dict:
    """Update the value and volatility parents of a continuous node.

    If the parents have value and/or volatility parents, they will be updated
    recursively.

    Updating the node's parents is a two step process:
        1. Update value parent(s) and their parents (if provided).
        2. Update volatility parent(s) and their parents (if provided).

    Then returns the new node structure.

    Parameters
    ----------
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
        .. note::
           `"psis"` is the value coupling strength. It should have same length than the
           volatility parents' indexes.
           `"kappas"` is the volatility coupling strength. It should have same length
           than the volatility parents' indexes.
    time_step :
        Interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that need to be updated. After continuous update, the
        parameters of value and volatility parents (if any) will be different.
    node_structure :
        Tuple of :py:class:`pyhgf.typing.Indexes` with same length than number of node.
        For each node, the index list value and volatility parents.

    Returns
    -------
    parameters_structure :
        The updated node structure.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    """
    # using the current node index, unwrap parameters and parents
    node_parameters = parameters_structure[node_idx]
    value_parents_idx = node_structure[node_idx].value_parents
    volatility_parents_idx = node_structure[node_idx].volatility_parents

    # return here if no parents node are provided
    if (value_parents_idx is None) and (volatility_parents_idx is None):
        return parameters_structure

    pihat = node_parameters["pihat"]

    #######################################
    # Update the continuous value parents #
    #######################################
    if value_parents_idx is not None:

        vape = node_parameters["mu"] - node_parameters["muhat"]
        psis = node_parameters["psis"]

        for va_pa_idx, psi in zip(value_parents_idx, psis):

            # unpack the current parent's parameters with value and volatility parents
            va_pa_node_parameters = parameters_structure[va_pa_idx]
            va_pa_value_parents_idx = node_structure[va_pa_idx].value_parents
            va_pa_volatility_parents_idx = node_structure[va_pa_idx].volatility_parents

            # Compute new value for nu and pihat
            logvol = va_pa_node_parameters["omega"]

            # Look at the (optional) va_pa's volatility parents
            # and update logvol accordingly
            if va_pa_volatility_parents_idx is not None:
                for va_pa_vo_pa, k in zip(
                    va_pa_volatility_parents_idx, va_pa_node_parameters["kappas"]
                ):
                    logvol += k * parameters_structure[va_pa_vo_pa]["mu"]

            # Estimate new_nu
            nu = time_step * jnp.exp(logvol)
            new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

            pihat_pa, nu_pa = [1 / (1 / va_pa_node_parameters["pi"] + new_nu), new_nu]
            pi_pa = pihat_pa + psi**2 * pihat

            # Compute new muhat
            driftrate = va_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            if va_pa_value_parents_idx is not None:
                for va_pa_va_pa in va_pa_value_parents_idx:
                    driftrate += psi * va_pa_va_pa["mu"]

            muhat_pa = va_pa_node_parameters["mu"] + time_step * driftrate
            mu_pa = muhat_pa + psi * pihat / pi_pa * vape

            # Update this parent's parameters
            parameters_structure[va_pa_idx]["pihat"] = pihat_pa
            parameters_structure[va_pa_idx]["pi"] = pi_pa
            parameters_structure[va_pa_idx]["muhat"] = muhat_pa
            parameters_structure[va_pa_idx]["mu"] = mu_pa
            parameters_structure[va_pa_idx]["nu"] = nu_pa

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idx is not None:

        nu = node_parameters["nu"]
        kappas = node_parameters["kappas"]
        vope = (
            1 / node_parameters["pi"]
            + (node_parameters["mu"] - node_parameters["muhat"]) ** 2
        ) * node_parameters["pihat"] - 1

        for vo_pa_idx, kappa in zip(volatility_parents_idx, kappas):

            # unpack the current parent's parameters with value and volatility parents
            vo_pa_node_parameters = parameters_structure[vo_pa_idx]
            vo_pa_value_parents_idx = node_structure[vo_pa_idx].value_parents
            vo_pa_volatility_parents_idx = node_structure[vo_pa_idx].volatility_parents

            # Compute new value for nu and pihat
            logvol = vo_pa_node_parameters["omega"]

            # Look at the (optional) vo_pa's volatility parents
            # and update logvol accordingly
            if vo_pa_volatility_parents_idx is not None:
                for i, vo_pa_vo_pa in enumerate(vo_pa_volatility_parents_idx):
                    k = vo_pa_node_parameters["kappas"][i]
                    logvol += k * parameters_structure[vo_pa_vo_pa]["mu"]

            # Estimate new_nu
            new_nu = time_step * jnp.exp(logvol)
            new_nu = jnp.where(new_nu > 1e-128, new_nu, jnp.nan)

            pihat_pa, nu_pa = [1 / (1 / vo_pa_node_parameters["pi"] + new_nu), new_nu]

            pi_pa = pihat_pa + 0.5 * (kappa * nu * pihat) ** 2 * (
                1 + (1 - 1 / (nu * node_parameters["pi"])) * vope
            )
            pi_pa = jnp.where(pi_pa <= 0, jnp.nan, pi_pa)

            # Compute new muhat
            driftrate = vo_pa_node_parameters["rho"]

            # Look at the (optional) va_pa's value parents
            # and update driftrate accordingly
            if vo_pa_value_parents_idx is not None:
                for vo_pa_va_pa in vo_pa_value_parents_idx:
                    driftrate += psi * parameters_structure[vo_pa_va_pa]["mu"]

            muhat_pa = vo_pa_node_parameters["mu"] + time_step * driftrate
            mu_pa = muhat_pa + 0.5 * kappa * nu * pihat / pi_pa * vope

            # Update this parent's parameters
            parameters_structure[vo_pa_idx]["pihat"] = pihat_pa
            parameters_structure[vo_pa_idx]["pi"] = pi_pa
            parameters_structure[vo_pa_idx]["muhat"] = muhat_pa
            parameters_structure[vo_pa_idx]["mu"] = mu_pa
            parameters_structure[vo_pa_idx]["nu"] = nu_pa

    return parameters_structure


@partial(jit, static_argnames=("node_structure", "node_idx"))
def continuous_input_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: Tuple,
    value: float,
) -> Dict:
    """Update the input node structure.

    This function is the entry level of the model fitting. It update the partents of
    the input node and then call :py:func:`pyhgf.jax.continuous_node_update` recursively
    to update the rest of the node structure.

    Parameters
    ----------
    value :
        The new observed value.
    time_step :
        Interval between the previous time point and the current time point.
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
        .. note::
           `"psis"` is the value coupling strength. It should have same length than the
           volatility parents' indexes.
           `"kappas"` is the volatility coupling strength. It should have same length
           than the volatility parents' indexes.
    node_structure :
        Tuple of :py:class:`pyhgf.typing.Indexes` with same length than number of node.
        For each node, the index list value and volatility parents.
    node_idx :
        Pointer to the node that need to be updated. After continuous update, the
        parameters of value and volatility parents (if any) will be different.

    Returns
    -------
    parameters_structure :
        The updated parameters structure.

    See Also
    --------
    continuous_node_update, update_binary_input_parents

    """
    # using the current node index, unwrap parameters and parents
    input_node_parameters = parameters_structure[node_idx]
    value_parents_idx = node_structure[node_idx].value_parents
    volatility_parents_idx = node_structure[node_idx].volatility_parents

    lognoise = input_node_parameters["omega"]

    if volatility_parents_idx is not None:
        for i, vo_pa_idx in enumerate(volatility_parents_idx):
            lognoise += (
                input_node_parameters["kappas"][i]
                * parameters_structure[vo_pa_idx]["mu"]
            )

    pihat = 1 / jnp.exp(lognoise)

    ########################
    # Update value parents #
    ########################
    if value_parents_idx is not None:

        # unpack the current parent's parameters with value and volatility parents
        va_pa_node_parameters = parameters_structure[value_parents_idx[0]]
        va_pa_value_parents_idx = node_structure[value_parents_idx[0]].value_parents
        va_pa_volatility_parents_idx = node_structure[
            value_parents_idx[0]
        ].volatility_parents

        vape = va_pa_node_parameters["mu"] - va_pa_node_parameters["muhat"]

        # Compute new value for nu and pihat
        logvol = va_pa_node_parameters["omega"]

        # Look at the (optional) va_pa's volatility parents
        # and update logvol accordingly
        if va_pa_volatility_parents_idx is not None:
            for va_pa_vo_pa, k in zip(
                va_pa_volatility_parents_idx, va_pa_node_parameters["kappas"]
            ):
                logvol += k * parameters_structure[va_pa_vo_pa]["mu"]

        # Estimate new_nu
        nu = time_step * jnp.exp(logvol)
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
        if va_pa_value_parents_idx is not None:
            for va_pa_va_pa in va_pa_value_parents_idx:
                driftrate += (
                    va_pa_node_parameters["psis"]
                    * parameters_structure[va_pa_va_pa]["mu"]
                )

        muhat_va_pa = va_pa_node_parameters["mu"] + time_step * driftrate
        vape = value - muhat_va_pa
        mu_va_pa = muhat_va_pa + pihat / pi_va_pa * vape

        # update input node's parameters
        parameters_structure[value_parents_idx[0]]["pihat"] = pihat_va_pa
        parameters_structure[value_parents_idx[0]]["pi"] = pi_va_pa
        parameters_structure[value_parents_idx[0]]["muhat"] = muhat_va_pa
        parameters_structure[value_parents_idx[0]]["mu"] = mu_va_pa
        parameters_structure[value_parents_idx[0]]["nu"] = nu_va_pa

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idx is not None:

        # unpack the current parent's parameters with value and volatility parents
        vo_pa_node_parameters = parameters_structure[volatility_parents_idx[0]]
        vo_pa_value_parents_idx = node_structure[
            volatility_parents_idx[0]
        ].value_parents
        vo_pa_volatility_parents_idx = node_structure[
            volatility_parents_idx[0]
        ].volatility_parents

        vope = (1 / pi_va_pa + (value - mu_va_pa) ** 2) * pihat - 1

        # Compute new value for nu and pihat
        logvol = vo_pa_node_parameters["omega"]

        # Look at the (optional) vo_pa's volatility parents
        # and update logvol accordingly
        if vo_pa_volatility_parents_idx is not None:
            for vo_pa_vo_pa in vo_pa_volatility_parents_idx:
                kappa = vo_pa_vo_pa[0]["kappas"]
                logvol += kappa * parameters_structure[vo_pa_vo_pa]["mu"]

        # Estimate new_nu
        nu = time_step * jnp.exp(logvol)
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
        if vo_pa_value_parents_idx is not None:
            for vo_pa_va_pa, p in zip(
                vo_pa_value_parents_idx, vo_pa_node_parameters["psis"]
            ):
                driftrate += p * parameters_structure[vo_pa_va_pa]["mu"]

        muhat_vo_pa = vo_pa_node_parameters["mu"] + time_step * driftrate
        mu_vo_pa = (
            muhat_vo_pa
            + jnp.multiply(0.5, input_node_parameters["kappas"]) / pi_vo_pa * vope
        )

        # Update this parent's parameters
        parameters_structure[volatility_parents_idx[0]]["pihat"] = pihat_vo_pa
        parameters_structure[volatility_parents_idx[0]]["pi"] = pi_vo_pa
        parameters_structure[volatility_parents_idx[0]]["muhat"] = muhat_vo_pa
        parameters_structure[volatility_parents_idx[0]]["mu"] = mu_vo_pa
        parameters_structure[volatility_parents_idx[0]]["nu"] = nu_vo_pa

    # store input surprise, value and timestep in the node's parameters
    parameters_structure[node_idx]["surprise"] = gaussian_surprise(
        x=value, muhat=muhat_va_pa, pihat=pihat
    )
    parameters_structure[node_idx]["time_step"] = time_step
    parameters_structure[node_idx]["value"] = value

    return parameters_structure


def gaussian_surprise(
    x: jnp.DeviceArray, muhat: jnp.DeviceArray, pihat: jnp.DeviceArray
) -> jnp.DeviceArray:
    r"""Surprise at an outcome under a Gaussian prediction.

    The surprise ellicited by an observation :math:`x` under a Gaussian distribution
    with expected mean :math:`\hat{\mu}` and expected precision :math:`\hat{\pi}` is
    given by:

    .. math::

       \frac{1}{2} \log(2 \pi) - \log(\hat{\pi}) + \hat{\pi}\sqrt{x - \hat{\mu}}

    where :math:`\pi` is the mathematical constant.

    Parameters
    ----------
    x :
        The outcome.
    muhat :
        The expected mean of the Gaussian distribution.
    pihat :
        The expected precision of the Gaussian distribution.

    Returns
    -------
    surprise :
        The Gaussian surprise.

    Examples
    --------
    >>> from pyhgf.continuous import gaussian_surprise
    >>> gaussian_surprise(x=2.0, muhat=0.0, pihat=1.0)
    `Array(2.9189386, dtype=float32, weak_type=True)`

    """
    return jnp.array(0.5) * (
        jnp.log(jnp.array(2.0) * jnp.pi)
        - jnp.log(pihat)
        + pihat * jnp.square(jnp.subtract(x, muhat))
    )
