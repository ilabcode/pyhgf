# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Union

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from pyhgf.typing import NodeStructure


@partial(jit, static_argnames=("node_structure", "node_idx"))
def continuous_node_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: NodeStructure,
    **args
) -> Dict:
    """Update the value and volatility parent(s) of a continuous node.

    Updating the node's parents is a two-step process:
    1. Update value parent(s).
    2. Update volatility parent(s).

    If a value/volatility parent has multiple children, all the children will update
    the parent together, therefor this function should only be called once per group
    of child nodes. The method :py:meth:`pyhgf.model.HGF.get_update_sequence`
    ensures that this function is only called once all the children have been
    updated.

    Then returns the structure of the new parameters.

    Parameters
    ----------
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    node_structure :
        Tuple of :py:class:`pyhgf.typing.Indexes` with the same length as node number.
        For each node, the index list value and volatility parents.

    Returns
    -------
    parameters_structure :
        The updated parameters structure.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # list value and volatility parents
    value_parents_idxs = node_structure[node_idx].value_parents
    volatility_parents_idxs = node_structure[node_idx].volatility_parents

    # return here if no parents node are provided
    if (value_parents_idxs is None) and (volatility_parents_idxs is None):
        return parameters_structure

    pihat = parameters_structure[node_idx]["pihat"]

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        # the strength of the value coupling between the base node and the parents nodes
        psis = parameters_structure[node_idx]["psis_parents"]

        for value_parent_idx, psi in zip(value_parents_idxs, psis):
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if node_structure[value_parent_idx].value_children[-1] == node_idx:
                # list the value and volatility parents
                value_parent_value_parents_idxs = node_structure[
                    value_parent_idx
                ].value_parents
                value_parent_volatility_parents_idxs = node_structure[
                    value_parent_idx
                ].volatility_parents

                # Compute new value for nu and pihat
                logvol = parameters_structure[value_parent_idx]["omega"]

                # Look at the (optional) va_pa's volatility parents
                # and update logvol accordingly
                if value_parent_volatility_parents_idxs is not None:
                    for value_parent_volatility_parents_idx, k in zip(
                        value_parent_volatility_parents_idxs,
                        parameters_structure[value_parent_idx]["kappas_parents"],
                    ):
                        logvol += (
                            k
                            * parameters_structure[value_parent_volatility_parents_idx][
                                "mu"
                            ]
                        )

                # Estimate new_nu
                nu = time_step * jnp.exp(logvol)
                new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

                pihat_value_parent, nu_value_parent = [
                    1 / (1 / parameters_structure[value_parent_idx]["pi"] + new_nu),
                    new_nu,
                ]

                # gather precision updates from other nodes if the parent has many
                # children - this part corresponds to the sum over children
                # required for the multi-children situations
                pi_children = 0.0
                for child_idx, psi_child in zip(
                    node_structure[value_parent_idx].value_children,
                    parameters_structure[value_parent_idx]["psis_children"],
                ):
                    pihat_child = parameters_structure[child_idx]["pihat"]
                    pi_children += psi_child**2 * pihat_child

                pi_value_parent = pihat_value_parent + pi_children

                # Compute new muhat
                driftrate = parameters_structure[value_parent_idx]["rho"]

                # Look at the (optional) va_pa's value parents
                # and update drift rate accordingly
                if value_parent_value_parents_idxs is not None:
                    for va_pa_va_pa in value_parent_value_parents_idxs:
                        driftrate += psi * parameters_structure[va_pa_va_pa]["mu"]

                muhat_value_parent = (
                    parameters_structure[value_parent_idx]["mu"] + time_step * driftrate
                )

                # gather PE updates from other nodes if the parent has many children
                # this part corresponds to the sum of children required for the
                # multi-children situations
                pe_children = 0.0
                for child_idx, psi_child in zip(
                    node_structure[value_parent_idx].value_children,
                    parameters_structure[value_parent_idx]["psis_children"],
                ):
                    vape_child = (
                        parameters_structure[child_idx]["mu"]
                        - parameters_structure[child_idx]["muhat"]
                    )
                    pihat_child = parameters_structure[child_idx]["pihat"]
                    pe_children += (
                        psi_child * pihat_child * vape_child
                    ) / pi_value_parent

                mu_value_parent = muhat_value_parent + pe_children

                # Update this parent's parameters
                parameters_structure[value_parent_idx]["pihat"] = pihat_value_parent
                parameters_structure[value_parent_idx]["pi"] = pi_value_parent
                parameters_structure[value_parent_idx]["muhat"] = muhat_value_parent
                parameters_structure[value_parent_idx]["mu"] = mu_value_parent
                parameters_structure[value_parent_idx]["nu"] = nu_value_parent

    #############################
    # Update volatility parents #
    #############################
    if volatility_parents_idxs is not None:
        # the strength of the value coupling between the base node and the parents nodes
        kappas = parameters_structure[node_idx]["kappas_parents"]

        nu = parameters_structure[node_idx]["nu"]
        vope = (
            1 / parameters_structure[node_idx]["pi"]
            + (
                parameters_structure[node_idx]["mu"]
                - parameters_structure[node_idx]["muhat"]
            )
            ** 2
        ) * parameters_structure[node_idx]["pihat"] - 1

        for volatility_parents_idx, kappa in zip(volatility_parents_idxs, kappas):
            # list the value and volatility parents
            volatility_parent_value_parents_idx = node_structure[
                volatility_parents_idx
            ].value_parents
            volatility_parent_volatility_parents_idx = node_structure[
                volatility_parents_idx
            ].volatility_parents

            # Compute new value for nu and pihat
            logvol = parameters_structure[volatility_parents_idx]["omega"]

            # Look at the (optional) vo_pa's volatility parents
            # and update logvol accordingly
            if volatility_parent_volatility_parents_idx is not None:
                for vo_pa_vo_pa, k in zip(
                    volatility_parent_volatility_parents_idx,
                    parameters_structure[volatility_parents_idx]["kappas_parents"],
                ):
                    logvol += k * parameters_structure[vo_pa_vo_pa]["mu"]

            # Estimate new_nu
            new_nu = time_step * jnp.exp(logvol)
            new_nu = jnp.where(new_nu > 1e-128, new_nu, jnp.nan)

            pihat_volatility_parent, nu_volatility_parent = [
                1 / (1 / parameters_structure[volatility_parents_idx]["pi"] + new_nu),
                new_nu,
            ]

            pi_volatility_parent = pihat_volatility_parent + 0.5 * (
                kappa * nu * pihat
            ) ** 2 * (1 + (1 - 1 / (nu * parameters_structure[node_idx]["pi"])) * vope)
            pi_volatility_parent = jnp.where(
                pi_volatility_parent <= 0, jnp.nan, pi_volatility_parent
            )

            # Compute new muhat
            driftrate = parameters_structure[volatility_parents_idx]["rho"]

            # Look at the (optional) va_pa's value parents
            # and update drift rate accordingly
            if volatility_parent_value_parents_idx is not None:
                for vo_pa_va_pa in volatility_parent_value_parents_idx:
                    driftrate += psi * parameters_structure[vo_pa_va_pa]["mu"]

            muhat_volatility_parent = (
                parameters_structure[volatility_parents_idx]["mu"]
                + time_step * driftrate
            )
            mu_volatility_parent = (
                muhat_volatility_parent
                + 0.5 * kappa * nu * pihat / pi_volatility_parent * vope
            )

            # Update this parent's parameters
            parameters_structure[volatility_parents_idx][
                "pihat"
            ] = pihat_volatility_parent
            parameters_structure[volatility_parents_idx]["pi"] = pi_volatility_parent
            parameters_structure[volatility_parents_idx][
                "muhat"
            ] = muhat_volatility_parent
            parameters_structure[volatility_parents_idx]["mu"] = mu_volatility_parent
            parameters_structure[volatility_parents_idx]["nu"] = nu_volatility_parent

    return parameters_structure


@partial(jit, static_argnames=("node_structure", "node_idx"))
def continuous_input_update(
    parameters_structure: Dict,
    time_step: float,
    node_idx: int,
    node_structure: NodeStructure,
    value: float,
) -> Dict:
    """Update the input node structure.

    This function is the entry-level of the structure updates. It updates the parent
    of the input node.

    Parameters
    ----------
    value :
        The new observed value.
    time_step :
        The interval between the previous time point and the current time point.
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "omega"` for
        continuous nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
    node_structure :
        Tuple of :py:class:`pyhgf.typing.Indexes` with same length than number of node.
        For each node, the index list value and volatility parents.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.

    Returns
    -------
    parameters_structure :
        The updated parameters structure.

    See Also
    --------
    continuous_node_update, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and timestep in the node's parameters
    parameters_structure[node_idx]["time_step"] = time_step
    parameters_structure[node_idx]["value"] = value

    # list value and volatility parents
    value_parents_idxs = node_structure[node_idx].value_parents

    ########################
    # Update value parents #
    ########################
    if value_parents_idxs is not None:
        for value_parent_idx in value_parents_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if node_structure[value_parent_idx].value_children[-1] == node_idx:
                # list value and volatility parents
                value_parent_value_parents_idxs = node_structure[
                    value_parent_idx
                ].value_parents
                value_parent_volatility_parents_idxs = node_structure[
                    value_parent_idx
                ].volatility_parents

                # Compute new value for nu and pihat
                logvol = parameters_structure[value_parent_idx]["omega"]

                # Look at the (optional) va_pa's volatility parents
                # and update logvol accordingly
                if value_parent_volatility_parents_idxs is not None:
                    for value_parent_volatility_parents_idx, k in zip(
                        value_parent_volatility_parents_idxs,
                        parameters_structure[value_parent_idx]["kappas_parents"],
                    ):
                        logvol += (
                            k
                            * parameters_structure[value_parent_volatility_parents_idx][
                                "mu"
                            ]
                        )

                # Estimate new_nu
                nu = time_step * jnp.exp(logvol)
                new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

                pihat_value_parent, nu_value_parent = [
                    1 / (1 / parameters_structure[value_parent_idx]["pi"] + new_nu),
                    new_nu,
                ]

                # gather precisions updates from other input nodes
                # in the case of a multivariate descendency
                pi_children = 0.0
                for child_idx, psi_child in zip(
                    node_structure[value_parent_idx].value_children,
                    parameters_structure[value_parent_idx]["psis_children"],
                ):
                    pihat_child = parameters_structure[child_idx]["pihat"]
                    pi_children += psi_child**2 * pihat_child

                pi_value_parent = pihat_value_parent + pi_children

                # Compute new muhat
                driftrate = parameters_structure[value_parent_idx]["rho"]

                # Look at the (optional) va_pa's value parents
                # and update drift rate accordingly
                if value_parent_value_parents_idxs is not None:
                    for (
                        value_parent_value_parents_idx
                    ) in value_parent_value_parents_idxs:
                        driftrate += (
                            parameters_structure[value_parent_idx]["psis_parents"][0]
                            * parameters_structure[value_parent_value_parents_idx]["mu"]
                        )

                muhat_value_parent = (
                    parameters_structure[value_parent_idx]["mu"] + time_step * driftrate
                )

                # gather PE updates from other input nodes
                # in the case of a multivariate descendency
                pe_children = 0.0
                for child_idx, psi_child in zip(
                    node_structure[value_parent_idx].value_children,
                    parameters_structure[value_parent_idx]["psis_children"],
                ):
                    child_value_prediction_error = (
                        parameters_structure[child_idx]["value"] - muhat_value_parent
                    )
                    pihat_child = parameters_structure[child_idx]["pihat"]
                    pe_children += (
                        psi_child * pihat_child * child_value_prediction_error
                    ) / pi_value_parent

                mu_value_parent = muhat_value_parent + pe_children

                # update input node's parameters
                parameters_structure[value_parent_idx]["pihat"] = pihat_value_parent
                parameters_structure[value_parent_idx]["pi"] = pi_value_parent
                parameters_structure[value_parent_idx]["muhat"] = muhat_value_parent
                parameters_structure[value_parent_idx]["mu"] = mu_value_parent
                parameters_structure[value_parent_idx]["nu"] = nu_value_parent

    return parameters_structure


def gaussian_surprise(
    x: Union[float, ArrayLike],
    muhat: Union[float, ArrayLike],
    pihat: Union[float, ArrayLike],
) -> Array:
    r"""Surprise at an outcome under a Gaussian prediction.

    The surprise elicited by an observation :math:`x` under a Gaussian distribution
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
