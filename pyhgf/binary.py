# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Tuple

import jax.numpy as jnp
from jax import jit
from jax.lax import cond


@partial(jit, static_argnames=("node_structure", "node_idx"))
def binary_node_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: Tuple,
    **args
) -> Dict:
    """Update the value and volatility parents of a binary node.

    If the parents have value and/or volatility parents, they will be updated
    recursively.

    Updating the node's parents is a two step process:
        1. Update value parent(s) and their parents (if provided).
        2. Update volatility parent(s) and their parents (if provided).

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

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

    """
    # using the current node index, unwrap parameters and parents
    node_parameters = parameters_structure[node_idx]
    value_parents_idx = node_structure[node_idx].value_parents
    volatility_parents_idx = node_structure[node_idx].volatility_parents

    # Return here if no parents node are provided
    if (value_parents_idx is None) and (volatility_parents_idx is None):
        return parameters_structure

    pihat = node_parameters["pihat"]
    vape = jnp.subtract(node_parameters["mu"], node_parameters["muhat"])

    #######################################
    # Update the continuous value parents #
    #######################################
    if value_parents_idx is not None:

        # unpack the current parent's parameters with value and volatility parents
        va_pa_node_parameters = parameters_structure[value_parents_idx[0]]
        # va_pa_value_parents_idx = node_structure[value_parents_idx[0]].value_parents
        va_pa_volatility_parents_idx = node_structure[
            value_parents_idx[0]
        ].volatility_parents

        # 1. get pihat_pa and nu_pa from the value parent (x2)

        # 1.1 get new_nu (x2)
        # 1.1.1 get logvol
        logvol = va_pa_node_parameters["omega"]

        # 1.1.2 Look at the (optional) va_pa's volatility parents
        # and update logvol accordingly
        if va_pa_volatility_parents_idx is not None:
            for va_pa_vo_pa, k in zip(
                va_pa_volatility_parents_idx, va_pa_node_parameters["kappas"]
            ):
                logvol += k * parameters_structure[va_pa_vo_pa]["mu"]

        # 1.1.3 Compute new_nu
        nu = time_step * jnp.exp(logvol)
        new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

        # 1.2 Compute new value for nu and pihat
        pihat_pa, nu_pa = [1 / (1 / va_pa_node_parameters["pi"] + new_nu), new_nu]

        # 2.
        pi_pa = pihat_pa + 1 / pihat

        # 3. get muhat_pa from value parent (x2)

        # 3.1
        driftrate = va_pa_node_parameters["rho"]

        # TODO: this will be decided once we figure out the multi parents/children
        # 3.2 Look at the (optional) va_pa's value parents
        # and update driftrate accordingly
        # if va_pa_value_parents is not None:
        #     psi: DeviceArray
        #     for va_pa_va_pa, psi in zip(
        #         va_pa_value_parents, va_pa_node_parameters["psis"]
        #     ):
        #         _mu: DeviceArray = va_pa_va_pa[0]["mu"]
        #         driftrate += psi * _mu
        # 3.3
        muhat_pa = va_pa_node_parameters["mu"] + time_step * driftrate

        # 4.
        mu_pa = muhat_pa + vape / pi_pa  # This line differs from the continuous input

        # 5. Update node's parameters and node's parents recursively
        parameters_structure[value_parents_idx[0]]["pihat"] = pihat_pa
        parameters_structure[value_parents_idx[0]]["pi"] = pi_pa
        parameters_structure[value_parents_idx[0]]["muhat"] = muhat_pa
        parameters_structure[value_parents_idx[0]]["mu"] = mu_pa
        parameters_structure[value_parents_idx[0]]["nu"] = nu_pa

    return parameters_structure


@partial(jit, static_argnames=("node_structure", "node_idx"))
def binary_input_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: Tuple,
    value: float,
) -> Dict:
    """Update the input node structure given one observation.

    This function is the entry level of the model fitting. It update the partents of
    the input node and then call :py:func:`pyhgf.binary.binary_node_update` to update
    the rest of the node structure.

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
    update_continuous_parents, update_continuous_input_parents

    """
    # using the current node index, unwrap parameters and parents
    input_node_parameters = parameters_structure[node_idx]
    value_parents_idx = node_structure[node_idx].value_parents
    volatility_parents_idx = node_structure[node_idx].volatility_parents

    if (value_parents_idx is None) and (volatility_parents_idx is None):
        return parameters_structure

    pihat = input_node_parameters["pihat"]

    ################################
    # Update parents (binary node) #
    ################################

    if value_parents_idx is not None:

        # unpack the current parent's parameters with value and volatility parents
        # va_pa_node_parameters = parameters_structure[value_parents_idx[0]]
        va_pa_value_parents_idx = node_structure[value_parents_idx[0]].value_parents
        # va_pa_volatility_parents_idx = node_structure[
        #     value_parents_idx[0]
        # ].volatility_parents

        # 1. Compute new muhat_pa and pihat_pa from binary node parent
        # ------------------------------------------------------------

        # 1.1 Compute new_muhat from continuous node parent (x2)

        # 1.1.1 get rho from the value parent of the binary node (x2)
        driftrate = parameters_structure[va_pa_value_parents_idx[0]]["rho"]

        # TODO: this will be decided once we figure out the multi parents/children
        # # 1.1.2 Look at the (optional) va_pa's value parents (x3)
        # # and update the driftrate accordingly
        # if va_pa_value_parents[0][1] is not None:
        #     for va_pa_va_pa in va_pa_value_parents[0][1]:
        #         # For each x2's value parents (optional)
        #         _psi: DeviceArray = va_pa_value_parents[0][0]["psis"]
        #         _mu: DeviceArray = va_pa_va_pa[0]["mu"]
        #         driftrate += _psi * _mu

        # 1.1.3 compute new_muhat
        muhat_va_pa = (
            parameters_structure[va_pa_value_parents_idx[0]]["mu"]
            + time_step * driftrate
        )

        muhat_va_pa = sgm(muhat_va_pa)
        pihat_va_pa = 1 / (muhat_va_pa * (1 - muhat_va_pa))

        # 2. Compute surprise
        # -------------------
        eta0 = input_node_parameters["eta0"]
        eta1 = input_node_parameters["eta1"]

        mu_va_pa, pi_va_pa, surprise = cond(
            pihat == jnp.inf,
            input_surprise_inf,
            input_surprise_reg,
            (pihat, value, eta1, eta0, muhat_va_pa),
        )

        # Update value parent's parameters
        parameters_structure[value_parents_idx[0]]["pihat"] = pihat_va_pa
        parameters_structure[value_parents_idx[0]]["pi"] = pi_va_pa
        parameters_structure[value_parents_idx[0]]["muhat"] = muhat_va_pa
        parameters_structure[value_parents_idx[0]]["mu"] = mu_va_pa

    parameters_structure[node_idx]["surprise"] = surprise
    parameters_structure[node_idx]["time_step"] = time_step
    parameters_structure[node_idx]["value"] = value

    return parameters_structure


def gaussian_density(x: float, mu: float, pi: float) -> jnp.DeviceArray:
    """Gaussian density as defined by mean and precision."""
    return (
        pi
        / jnp.sqrt(2 * jnp.pi)
        * jnp.exp(jnp.subtract(0, pi) / 2 * (jnp.subtract(x, mu)) ** 2)
    )


def sgm(
    x,
    lower_bound=jnp.array(0.0),
    upper_bound=jnp.array(1.0),
):
    """Logistic sigmoid function."""
    return jnp.subtract(upper_bound, lower_bound) / (1 + jnp.exp(-x)) + lower_bound


def binary_surprise(x: float, muhat: float) -> jnp.DeviceArray:
    r"""Surprise at a binary outcome.

    The surprise ellicited by a binary observation :math:`x` mean :math:`\hat{\mu}`
    and expected probability :math:`\hat{\pi}` is given by:

    .. math::

       \begin{cases}
            -\log(\hat{\mu}),& \text{if } x=1\\
            -\log(1 - \hat{\mu}), & \text{if } x=0\\
        \end{cases}

    Parameters
    ----------
    x : jnp.DeviceArray
        The outcome.
    muhat : jnp.DeviceArray
        The mean of the Bernouilli distribution.

    Returns
    -------
    surprise : jnp.DeviceArray
        The binary surprise.


    Examples
    --------
    >>> from pyhgf.binary import binary_surprise
    >>> binary_surprise(x=1.0, muhat=0.7)
    `Array(0.35667497, dtype=float32, weak_type=True)`

    """
    return jnp.where(x, -jnp.log(muhat), -jnp.log(jnp.array(1.0) - muhat))


def input_surprise_inf(op):
    """Apply special case if pihat is `jnp.inf` (just pass the value through)."""
    (pihat, value, eta1, eta0, muhat_va_pa) = op
    mu_va_pa = value
    pi_va_pa = jnp.inf
    surprise = binary_surprise(value, muhat_va_pa)

    return mu_va_pa, pi_va_pa, surprise


def input_surprise_reg(op):
    """Compute the surprise, mu_va_pa and pi_va_pa if pihat is not `jnp.inf`."""
    (pihat, value, eta1, eta0, muhat_va_pa) = op

    # Likelihood under eta1
    und1 = jnp.exp(jnp.subtract(0, pihat) / 2 * (jnp.subtract(value, eta1)) ** 2)

    # Likelihood under eta0
    und0 = jnp.exp(jnp.subtract(0, pihat) / 2 * (jnp.subtract(value, eta0)) ** 2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    mu_va_pa = muhat_va_pa * und1 / (muhat_va_pa * und1 + (1 - muhat_va_pa) * und0)
    pi_va_pa = 1 / (mu_va_pa * (1 - mu_va_pa))

    # Surprise
    surprise = -jnp.log(
        muhat_va_pa * gaussian_density(value, eta1, pihat)
        + (1 - muhat_va_pa) * gaussian_density(value, eta0, pihat)
    )

    return mu_va_pa, pi_va_pa, surprise
