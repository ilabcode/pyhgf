# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_mean_continuous_node(
    attributes: Dict, edges: Edges, node_idx: int, node_precision: float
) -> float:
    r"""Update the mean of a state node using the value prediction errors.

    1. Mean update from value coupling.

    The new mean of a state node :math:`b` value coupled with other input and/or state
    nodes :math:`j` at time :math:`k` is given by:

    .. math::
        \mu_b^{(k)} =  \hat{\mu}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \frac{\kappa_j \hat{\pi}_j^{(k)}}{\pi_b} \delta_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed beforehand. If the child node is an input node, this value was
    computed by
    :py:func:`pyhgf.updates.prediction_errors.inputs.continuous.continuous_input_value_prediction_error`.
    If the child node is a state node, this value was computed by
    :py:func:`pyhgf.updates.prediction_errors.nodes.continuous.continuous_node_value_prediction_error`.

    2. Mean update from volatility coupling.

    The new mean of a state node :math:`b` volatility coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    .. math::
        \mu_b^{(k)} = \hat{\mu}_b^{(k)} + \frac{1}{2\pi_b}
          \sum_{j=1}^{N_{children}} \kappa_j \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error is given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    with :math:`\Omega_j^{(k)}` the predicted volatility computed in the prediction
    step :py:func:`pyhgf.updates.prediction.predict_precision`.

    If the child node is an input node, the volatility prediction error
    :math:`\Delta_j^{(k)}` was computed by
    :py:func:`pyhgf.updates.prediction_errors.inputs.continuous.continuous_input_volatility_prediction_error`.
    If the child node is a state node, this value was computed by
    :py:func:`pyhgf.updates.prediction_errors.nodes.continuous.continuous_node_volatility_prediction_error`.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    node_idx :
        Pointer to the value parent node that will be updated.
    node_precision :
        The precision of the node. Depending on the kind of volatility update, this
        value can be the expected precision (ehgf), or the posterior from the update
        (standard).

    Returns
    -------
    posterior_mean :
        The new posterior mean.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    posterior_update_precision_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # sum the prediction errors from both value and volatility coupling
    (
        value_precision_weigthed_prediction_error,
        volatility_precision_weigthed_prediction_error,
    ) = (0.0, 0.0)

    # Value coupling updates - update the mean of a value parent
    # ----------------------------------------------------------
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
        ):
            # get the value prediction error (VAPE)
            # if this is jnp.nan (no observation) set the VAPE to 0.0
            value_prediction_error = attributes[value_child_idx]["temp"][
                "value_prediction_error"
            ]

            # cancel the prediction error if the child value was not observed
            value_prediction_error *= attributes[value_child_idx]["observed"]

            # expected precisions from the value children
            # sum the precision weigthed prediction errors over all children
            value_precision_weigthed_prediction_error += (
                (value_coupling * attributes[value_child_idx]["expected_precision"])
                / node_precision
            ) * value_prediction_error

    # Volatility coupling updates - update the mean of a volatility parent
    # --------------------------------------------------------------------
    if edges[node_idx].volatility_children is not None:
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # get the volatility prediction error (VOPE)
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]

            # retrieve the effective precision (γ) computed during the prediction step
            effective_precision = attributes[volatility_child_idx]["temp"][
                "effective_precision"
            ]

            # the precision weigthed prediction error
            precision_weigthed_prediction_error = (
                volatility_coupling * effective_precision * volatility_prediction_error
            )

            # weight using the node's precision
            precision_weigthed_prediction_error *= 1 / (2 * node_precision)

            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error *= attributes[volatility_child_idx][
                "observed"
            ]

            # aggregate over volatility children
            volatility_precision_weigthed_prediction_error += (
                precision_weigthed_prediction_error
            )

    # Compute the new posterior mean
    # using value prediction errors from both value and volatility coupling
    posterior_mean = (
        attributes[node_idx]["expected_mean"]
        + value_precision_weigthed_prediction_error
        + volatility_precision_weigthed_prediction_error
    )

    return posterior_mean


@partial(jit, static_argnames=("edges", "node_idx"))
def posterior_update_precision_continuous_node(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
    time_step: float,
) -> float:
    r"""Update the precision of a state node using the volatility prediction errors.

    #. Precision update from value coupling.

    The new precision of a state node :math:`b` value coupled with other input and/or
    state nodes :math:`j` at time :math:`k` is given by:

    .. math::

            \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
            \kappa_j^2 \hat{\pi}_j^{(k)}

    Where :math:`\kappa_j` is the volatility coupling strength between the child node
    and the state node and :math:`\delta_j^{(k)}` is the value prediction error that
    was computed before hand. If the child node is an input node, this value was
    computed by
    :py:func:`pyhgf.updates.prediction_errors.inputs.continuous.continuous_input_value_prediction_error`.
    If the child node is a state node, this value was computed by
    :py:func:`pyhgf.updates.prediction_errors.nodes.continuous.continuous_node_value_prediction_error`.


    #. Precision update from volatility coupling.

    The new precision of a state node :math:`b` volatility coupled with other input
    and/or state nodes :math:`j` at time :math:`k` is given by:

    .. math::

        \pi_b^{(k)} = \hat{\pi}_b^{(k)} + \sum_{j=1}^{N_{children}}
        \frac{1}{2} \left( \kappa_j \gamma_j^{(k)} \right) ^2 +
        \left( \kappa_j \gamma_j^{(k)} \right) ^2 \Delta_j^{(k)} -
        \frac{1}{2} \kappa_j^2 \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the effective precision of the prediction, given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    that was computed in the prediction step.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    node_idx :
        Pointer to the value parent node that will be updated.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    posterior_precision :
        The new posterior precision.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # sum the prediction errors from both value and volatility coupling
    precision_weigthed_prediction_error = 0.0

    # Value coupling updates - update the precision of a value parent
    # ---------------------------------------------------------------
    if edges[node_idx].value_children is not None:
        for value_child_idx, value_coupling in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
        ):
            precision_weigthed_prediction_error += (
                value_coupling**2 * attributes[value_child_idx]["expected_precision"]
            )

            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error *= attributes[value_child_idx][
                "observed"
            ]

    # Volatility coupling updates - update the precision of a volatility parent
    # -------------------------------------------------------------------------
    if edges[node_idx].volatility_children is not None:
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # volatility weigthed precision for the volatility child (γ)
            effective_precision = attributes[volatility_child_idx]["temp"][
                "effective_precision"
            ]

            # retrieve the volatility prediction error
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]

            # sum over all volatility children
            precision_weigthed_prediction_error += (
                0.5 * (volatility_coupling * effective_precision) ** 2
                + (volatility_coupling * effective_precision) ** 2
                * volatility_prediction_error
                - 0.5
                * volatility_coupling**2
                * effective_precision
                * volatility_prediction_error
            )

            # cancel the prediction error if the child value was not observed
            precision_weigthed_prediction_error *= attributes[volatility_child_idx][
                "observed"
            ]

    # Compute the new posterior precision
    # using value prediction errors from both value and volatility coupling
    posterior_precision = (
        attributes[node_idx]["expected_precision"] + precision_weigthed_prediction_error
    )

    # ensure the new precision is greater than 0
    observed_posterior_precision = jnp.where(
        posterior_precision > 1e-128, posterior_precision, jnp.nan
    )

    # additionnal steps for unobserved values
    # ----------------------------------------------------------------------------------

    # List the node's volatility parents
    volatility_parents_idxs = edges[node_idx].volatility_parents

    # Get the tonic volatility from the node
    total_volatility = attributes[node_idx]["tonic_volatility"]

    # Look at the (optional) volatility parents and add their value to the tonic
    # volatility to get the total volatility
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            total_volatility += (
                volatility_coupling * attributes[volatility_parents_idx]["mean"]
            )

    # compute the predicted_volatility from the total volatility
    predicted_volatility = time_step * jnp.exp(total_volatility)

    # Estimate the new precision for the continuous state node
    unobserved_posterior_precision = 1 / (
        (1 / attributes[node_idx]["precision"]) + predicted_volatility
    )

    # for all children, look at the values of VAPE
    # if all these values are NaNs, the node has not received observations
    observations = []
    if edges[node_idx].value_children is not None:
        for children_idx in edges[node_idx].value_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    if edges[node_idx].volatility_children is not None:
        for children_idx in edges[node_idx].volatility_children:  # type: ignore
            observations.append(attributes[children_idx]["observed"])
    observations = jnp.any(jnp.array(observations))

    posterior_precision = (
        unobserved_posterior_precision * (1 - observations)  # type: ignore
        + observed_posterior_precision * observations
    )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_update(
    attributes: Dict, node_idx: int, edges: Edges, time_step: float, **args
) -> Dict:
    """Update the posterior of a continuous node using the standard HGF update.

    The standard HGF posterior update is a two-step process:
    1. Update the posterior precision.
    2. Update the posterior mean and assume that the posterior precision is the value
    updated in the first step.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    posterior_update_precision_continuous_node, posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # update the posterior mean and precision
    posterior_precision = posterior_update_precision_continuous_node(
        attributes, edges, node_idx, time_step
    )
    attributes[node_idx]["precision"] = posterior_precision

    posterior_mean = posterior_update_mean_continuous_node(
        attributes, edges, node_idx, node_precision=attributes[node_idx]["precision"]
    )
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_node_update_ehgf(
    attributes: Dict, node_idx: int, edges: Edges, time_step: float, **args
) -> Dict:
    """Update the posterior of a continuous node using the eHGF update.

    The eHGF posterior update is a two-step process:
    1. Update the posterior mean and assume that the posterior precision is equal to
    the expected precision.
    2. Update the posterior precision.

    .. note::
        By updating the mean first, and approximating the precision using the expected,
        precision, this update step often perform better than the regular update and
        limit the occurence of negative precision that cause the model to fail under
        some circumstances

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The time elapsed between this observation and the previous one.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    posterior_update_precision_continuous_node, posterior_update_mean_continuous_node

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # update the posterior mean and precision using the eHGF update step
    # we start with the mean update using the expected precision as an approximation
    posterior_mean = posterior_update_mean_continuous_node(
        attributes,
        edges,
        node_idx,
        node_precision=attributes[node_idx]["expected_precision"],
    )
    attributes[node_idx]["mean"] = posterior_mean

    posterior_precision = posterior_update_precision_continuous_node(
        attributes, edges, node_idx, time_step
    )
    attributes[node_idx]["precision"] = posterior_precision

    return attributes
