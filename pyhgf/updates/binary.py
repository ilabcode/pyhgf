# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Union

import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from jax.typing import ArrayLike

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_update(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the value parent(s) of a binary node.

    In a three-level HGF, this step will update the node :math:`x_2`.

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

    Parameters
    ----------
    attributes :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
    time_step :
        Interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        Tuple of :py:class:`pyhgf.typing.Indexes` with the same length as node number.
        For each node, the index list value and volatility parents.

    Returns
    -------
    attributes :
        The updated node structure.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # using the current node index, unwrap parameters and parents
    value_parent_idxs = edges[node_idx].value_parents

    # Return here if no parents node are found
    if value_parent_idxs is None:
        return attributes

    #############################################
    # Update the continuous value parents (x-2) #
    #############################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                value_parent_value_parent_idxs = edges[value_parent_idx].value_parents
                value_parent_volatility_parent_idxs = edges[
                    value_parent_idx
                ].volatility_parents

                # get logvolatility
                logvol = attributes[value_parent_idx]["omega"]

                # 1.1.2 Look at the (optional) va_pa's volatility parents
                # and update logvol accordingly
                if value_parent_volatility_parent_idxs is not None:
                    for value_parent_volatility_parent_idx, k in zip(
                        value_parent_volatility_parent_idxs,
                        attributes[value_parent_idx]["kappas_parents"],
                    ):
                        logvol += (
                            k * attributes[value_parent_volatility_parent_idx]["mu"]
                        )

                # 1.1.3 Compute new_nu
                nu = time_step * jnp.exp(logvol)
                new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

                # 1.2 Compute new value for nu and pihat
                pihat_value_parent, nu_value_parent = [
                    1 / (1 / attributes[value_parent_idx]["pi"] + new_nu),
                    new_nu,
                ]

                # 2.
                # gather precision updates from other binray input nodes
                # this part corresponds to the sum over children
                # required for the multi-children situations
                pi_children = 0.0
                for child_idx, psi_child in zip(
                    edges[value_parent_idx].value_children,
                    attributes[value_parent_idx]["psis_children"],
                ):
                    pihat_child = attributes[child_idx]["pihat"]
                    pi_children += psi_child * (1 / pihat_child)

                pi_value_parent = pihat_value_parent + pi_children

                # 3. get muhat_value_parent from value parent (x2)

                # 3.1
                driftrate = attributes[value_parent_idx]["rho"]

                # 3.2 Look at the (optional) value parent's value parents
                # and update driftrate accordingly
                if value_parent_value_parent_idxs is not None:
                    for value_parent_value_parent_idx, psi in zip(
                        value_parent_value_parent_idxs,
                        attributes[value_parent_idx]["psis_parents"],
                    ):
                        driftrate += (
                            psi * attributes[value_parent_value_parent_idx]["mu"]
                        )

                # 3.3
                muhat_value_parent = (
                    attributes[value_parent_idx]["mu"] + time_step * driftrate
                )

                # gather PE updates from other binray child nodes if the parent has many
                # this part corresponds to the sum of children required for the
                # multi-children situations
                pe_children = 0.0
                for child_idx, psi_child in zip(
                    edges[value_parent_idx].value_children,
                    attributes[value_parent_idx]["psis_children"],
                ):
                    vape_child = (
                        attributes[child_idx]["mu"] - attributes[child_idx]["muhat"]
                    )
                    pe_children += (psi_child * vape_child) / pi_value_parent

                # 4.
                mu_value_parent = muhat_value_parent + pe_children

                # 5. Update node's parameters and node's parents recursively
                attributes[value_parent_idx]["pi"] = pi_value_parent
                attributes[value_parent_idx]["mu"] = mu_value_parent
                attributes[value_parent_idx]["nu"] = nu_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_node_prediction(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Update the value parent(s) of a binary node.

    In a three-level HGF, this step will update the node :math:`x_2`.

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

    Parameters
    ----------
    attributes :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
    time_step :
        Interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        Tuple of :py:class:`pyhgf.typing.Indexes` with the same length as node number.
        For each node, the index list value and volatility parents.

    Returns
    -------
    attributes :
        The updated node structure.

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # using the current node index, unwrap parameters and parents
    value_parent_idxs = edges[node_idx].value_parents

    # Return here if no parents node are found
    if value_parent_idxs is None:
        return attributes

    ############################################################
    # Update predictions in the continuous value parents (x-2) #
    ############################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # if this child is the last one relative to this parent's family, all the
            # children will update the parent at once, otherwise just pass and wait
            if edges[value_parent_idx].value_children[-1] == node_idx:
                value_parent_value_parent_idxs = edges[value_parent_idx].value_parents
                value_parent_volatility_parent_idxs = edges[
                    value_parent_idx
                ].volatility_parents

                # get log volatility
                logvol = attributes[value_parent_idx]["omega"]

                # look at the va_pa's volatility parents and update accordingly
                if value_parent_volatility_parent_idxs is not None:
                    for value_parent_volatility_parent_idx, k in zip(
                        value_parent_volatility_parent_idxs,
                        attributes[value_parent_idx]["kappas_parents"],
                    ):
                        logvol += (
                            k * attributes[value_parent_volatility_parent_idx]["mu"]
                        )

                # compute new_nu
                nu = time_step * jnp.exp(logvol)
                new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

                # compute new value for pihat
                pihat_value_parent = 1 / (
                    1 / attributes[value_parent_idx]["pi"] + new_nu
                )

                # drift rate
                driftrate = attributes[value_parent_idx]["rho"]

                # look at value parent's value parents and update driftrate accordingly
                if value_parent_value_parent_idxs is not None:
                    for value_parent_value_parent_idx, psi in zip(
                        value_parent_value_parent_idxs,
                        attributes[value_parent_idx]["psis_parents"],
                    ):
                        driftrate += (
                            psi * attributes[value_parent_value_parent_idx]["mu"]
                        )

                # compute new muhat
                muhat_value_parent = (
                    attributes[value_parent_idx]["mu"] + time_step * driftrate
                )

                # update the parent nodes' parameters
                attributes[value_parent_idx]["pihat"] = pihat_value_parent
                attributes[value_parent_idx]["muhat"] = muhat_value_parent

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_update(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Update the input node structure given one binary observation.

    This function is the entry-level of the binary node. It updates the parents of
    the input node (:math:`x_1`).

    Parameters
    ----------
    value :
        The new observed value.
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
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.

    Returns
    -------
    attributes :
        The updated parameters structure.

    See Also
    --------
    update_continuous_parents, update_continuous_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # list value and volatility parents
    value_parent_idxs = edges[node_idx].value_parents
    volatility_parent_idxs = edges[node_idx].volatility_parents

    if (value_parent_idxs is None) and (volatility_parent_idxs is None):
        return attributes

    pihat = attributes[node_idx]["pihat"]

    #######################################################
    # Update the value parent(s) of the binary input node #
    #######################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # list the (unique) value parents
            value_parent_value_parent_idxs = edges[value_parent_idx].value_parents[0]

            # 1. Compute new muhat_value_parent and pihat_value_parent
            # --------------------------------------------------------
            # 1.1 Compute new_muhat from continuous node parent (x2)
            # 1.1.1 get rho from the value parent of the binary node (x2)
            driftrate = attributes[value_parent_value_parent_idxs]["rho"]

            # # 1.1.2 Look at the (optional) value parent's value parents (x3)
            # # and update the drift rate accordingly
            if edges[value_parent_value_parent_idxs].value_parents is not None:
                for (
                    value_parent_value_parent_value_parent_idx,
                    psi_parent_parent,
                ) in zip(
                    edges[value_parent_value_parent_idxs].value_parents,
                    attributes[value_parent_value_parent_idxs]["psis_parents"],
                ):
                    # For each x2's value parents (optional)
                    driftrate += (
                        psi_parent_parent
                        * attributes[value_parent_value_parent_value_parent_idx]["mu"]
                    )

            # 1.1.3 compute new_muhat
            muhat_value_parent = (
                attributes[value_parent_value_parent_idxs]["mu"] + time_step * driftrate
            )

            muhat_value_parent = sgm(muhat_value_parent)

            # 2. Compute surprise
            # -------------------
            eta0 = attributes[node_idx]["eta0"]
            eta1 = attributes[node_idx]["eta1"]

            mu_value_parent, pi_value_parent, surprise = cond(
                pihat == jnp.inf,
                input_surprise_inf,
                input_surprise_reg,
                (pihat, value, eta1, eta0, muhat_value_parent),
            )

            # Update value parent's parameters
            attributes[value_parent_idx]["pi"] = pi_value_parent
            attributes[value_parent_idx]["mu"] = mu_value_parent

    attributes[node_idx]["surprise"] = surprise
    attributes[node_idx]["time_step"] = time_step
    attributes[node_idx]["value"] = value

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def binary_input_prediction(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Update the input node structure given one binary observation.

    This function is the entry-level of the binary node. It updates the parents of
    the input node (:math:`x_1`).

    Parameters
    ----------
    value :
        The new observed value.
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
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.

    Returns
    -------
    attributes :
        The updated parameters structure.

    See Also
    --------
    update_continuous_parents, update_continuous_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # list value and volatility parents
    value_parent_idxs = edges[node_idx].value_parents
    volatility_parent_idxs = edges[node_idx].volatility_parents

    if (value_parent_idxs is None) and (volatility_parent_idxs is None):
        return attributes

    #######################################################
    # Update the value parent(s) of the binary input node #
    #######################################################
    if value_parent_idxs is not None:
        for value_parent_idx in value_parent_idxs:
            # list the (unique) value parents
            value_parent_value_parent_idxs = edges[value_parent_idx].value_parents[0]

            # 1. Compute new muhat_value_parent and pihat_value_parent
            # --------------------------------------------------------
            # 1.1 Compute new_muhat from continuous node parent (x2)
            # 1.1.1 get rho from the value parent of the binary node (x2)
            driftrate = attributes[value_parent_value_parent_idxs]["rho"]

            # # 1.1.2 Look at the (optional) value parent's value parents (x3)
            # # and update the drift rate accordingly
            if edges[value_parent_value_parent_idxs].value_parents is not None:
                for (
                    value_parent_value_parent_value_parent_idx,
                    psi_parent_parent,
                ) in zip(
                    edges[value_parent_value_parent_idxs].value_parents,
                    attributes[value_parent_value_parent_idxs]["psis_parents"],
                ):
                    # For each x2's value parents (optional)
                    driftrate += (
                        psi_parent_parent
                        * attributes[value_parent_value_parent_value_parent_idx]["mu"]
                    )

            # 1.1.3 compute new_muhat
            muhat_value_parent = (
                attributes[value_parent_value_parent_idxs]["mu"] + time_step * driftrate
            )

            muhat_value_parent = sgm(muhat_value_parent)
            pihat_value_parent = 1 / (muhat_value_parent * (1 - muhat_value_parent))

            # Update value parent's parameters
            attributes[value_parent_idx]["pihat"] = pihat_value_parent
            attributes[value_parent_idx]["muhat"] = muhat_value_parent

    return attributes


def gaussian_density(x: ArrayLike, mu: ArrayLike, pi: ArrayLike) -> ArrayLike:
    """Gaussian density as defined by mean and precision."""
    return pi / jnp.sqrt(2 * jnp.pi) * jnp.exp(-pi / 2 * (x - mu) ** 2)


def sgm(
    x,
    lower_bound: Union[ArrayLike, float] = 0.0,
    upper_bound: Union[ArrayLike, float] = 1.0,
) -> ArrayLike:
    """Logistic sigmoid function."""
    return (upper_bound - lower_bound) / (1 + jnp.exp(-x)) + lower_bound


def binary_surprise(
    x: Union[float, ArrayLike], muhat: Union[float, ArrayLike]
) -> ArrayLike:
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
    x :
        The outcome.
    muhat :
        The mean of the Bernoulli distribution.

    Returns
    -------
    surprise :
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
    _, value, _, _, muhat_value_parent = op
    mu_value_parent = value
    pi_value_parent = jnp.inf
    surprise = binary_surprise(value, muhat_value_parent)

    return mu_value_parent, pi_value_parent, surprise


def input_surprise_reg(op):
    """Compute the surprise, mu_value_parent and pi_value_parent."""
    pihat, value, eta1, eta0, muhat_value_parent = op

    # Likelihood under eta1
    und1 = jnp.exp(-pihat / 2 * (value - eta1) ** 2)

    # Likelihood under eta0
    und0 = jnp.exp(-pihat / 2 * (value - eta0) ** 2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    mu_value_parent = (
        muhat_value_parent
        * und1
        / (muhat_value_parent * und1 + (1 - muhat_value_parent) * und0)
    )
    pi_value_parent = 1 / (mu_value_parent * (1 - mu_value_parent))

    # Surprise
    surprise = -jnp.log(
        muhat_value_parent * gaussian_density(value, eta1, pihat)
        + (1 - muhat_value_parent) * gaussian_density(value, eta0, pihat)
    )

    return mu_value_parent, pi_value_parent, surprise
