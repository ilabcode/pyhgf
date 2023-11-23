# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def posterior_update_mean_continuous_node(
    attributes: Dict, edges: Edges, node_idx: int, node_precision: float
) -> float:
    r"""Update the mean of a continuous node.

    Update the mean of the volatility parent.

    The new mean of the volatility parent :math:`a` of a state node at time :math:`k`
    is given by:

    .. math::
        \mu_a^{(k)} = \hat{\mu}_a^{(k)} + \frac{1}{2\pi_a}
          \sum_{j=1}^{N_{children}} \kappa_j \gamma_j^{(k)} \Delta_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\Delta_j^{(k)}` is the
    volatility prediction error given by:

    .. math::

        \Delta_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_j^{(k)}} +
        \hat{\pi}_j^{(k)} \left( \delta_j^{(k)} \right)^2 - 1

    with :math:`\delta_j^{(k)}` the value prediction error
    :math:`\delta_j^{(k)} = \mu_j^{k} - \hat{\mu}_j^{k}`.

    :math:`\gamma_j^{(k)}` is the volatility-weighted precision of the prediction,
    given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    with :math:`\Omega_j^{(k)}` the predicted volatility computed in the prediction
    step (:func:`pyhgf.updates.prediction.predict_precision`).



    The new mean of the volatility parent :math:`a` of an input node at time :math:`k`
    is given by:

    .. math::

        \mu_a^{(k)} = \hat{\mu}_a^{(k)} + \frac{1}{2\pi_a}
        \sum_{j=1}^{N_{children}} \kappa_j\epsilon_j^{(k)}

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\epsilon_j^{(k)}` is the
    noise prediction error given by:

    .. math::

        \epsilon_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_{vapa}^{(k)}} +
        \hat{\pi}_j^{(k)} \left( u^{(k)} - \mu_{vapa}^{(k)} \right)^2 - 1

    Note that, because we are working with continuous input nodes,
    :math:`\epsilon_j^{(k)}` is not a function of the value prediction error but uses
    the posterior of the value parent(s).

    The expected precision of the input is the sum of the tonic and phasic volatility,
    given by:

    .. math::

        \hat{\pi}_j^{(k)} = \frac{1}{\zeta} * \frac{1}{e^{\kappa_j \mu_a}}

    where :math:`\zeta` is the continuous input precision (in real space).


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
    node_precision :
        The precision of the node. Depending of the kind of volatility update, this
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
    # Value coupling updates
    # ----------------------
    if edges[node_idx].value_children is not None:
        precision_weigthed_prediction_error = 0.0
        for value_child_idx, value_coupling in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
        ):
            # prediction errors from the value children
            prediction_errors = (
                attributes[value_child_idx]["temp"]["value_prediction_error"]
                * value_coupling
            )

            # expected precisions from the value children
            precisions = (
                attributes[value_child_idx]["temp"]["expected_precision_children"]
                / attributes[node_idx]["expected_precision"]
            )

            # sum the precision weigthed over all children
            precision_weigthed_prediction_error += precisions * prediction_errors

        # Compute the new posterior mean
        posterior_mean = (
            attributes[node_idx]["expected_mean"] + precision_weigthed_prediction_error
        )

    # Volatility coupling updates
    # ---------------------------
    if edges[node_idx].volatility_children is not None:
        precision_weigthed_prediction_error = 0.0
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # retrieve the predicted volatility (Ω) computed in the prediction step
            predicted_volatility = attributes[volatility_child_idx]["temp"][
                "predicted_volatility"
            ]

            # volatility weigthed precision for the volatility child (γ)
            volatility_weigthed_precision = (
                attributes[volatility_child_idx]["expected_precision"]
                * predicted_volatility
            )

            # the precision weigthed prediction error
            precision_weigthed_prediction_error += (
                volatility_coupling
                * volatility_weigthed_precision
                * attributes[volatility_child_idx]["temp"][
                    "volatility_prediction_error"
                ]
            )

        # weight using the node's precision
        precision_weigthed_prediction_error *= 1 / (2 * node_precision)

        # Compute the new posterior mean
        posterior_mean = (
            attributes[node_idx]["expected_mean"] + precision_weigthed_prediction_error
        )

    return posterior_mean


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def posterior_update_precision_continuous_node(
    attributes: Dict,
    edges: Edges,
    node_idx: int,
) -> float:
    r"""Update the precision of a continuous node.

    The new precision of the volatility parent :math:`a` of a state node at time
    :math:`k` is given by:

    .. math::

        \pi_a^{(k)} = \hat{\pi}_a^{(k)} + \sum_{j=1}^{N_{children}}
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

    :math:`\gamma_j^{(k)}` is the volatility-weighted precision of the prediction,
    given by:

    .. math::

        \gamma_j^{(k)} = \Omega_j^{(k)} \hat{\pi}_j^{(k)}

    with :math:`\Omega_j^{(k)}` the predicted volatility computed in the prediction
    step (:func:`pyhgf.updates.prediction.predict_precision`).


    The new mean of the volatility parent :math:`a` of an input node at time :math:`k`
    is given by:

    .. math::

        \pi_a^{(k)} = \hat{\pi}_a^{(k)} + \sum_{j=1}^{N_{children}} \frac{1}{2}
         \kappa_j^2 \left( 1 + \epsilon_j^{(k)} \right)

    where :math:`\kappa_j` is the volatility coupling strength between the volatility
    parent and the volatility children :math:`j` and :math:`\epsilon_j^{(k)}` is the
    noise prediction error given by:

    .. math::

        \epsilon_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_{vapa}^{(k)}} +
        \hat{\pi}_j^{(k)} \left( u^{(k)} - \mu_{vapa}^{(k)} \right)^2 - 1

    Note that, because we are working with continuous input nodes,
    :math:`\epsilon_j^{(k)}` is not a function of the value prediction error but uses
    the posterior of the value parent(s).

    The expected precision of the input is the sum of the tonic and phasic volatility,
    given by:

    .. math::

        \hat{\pi}_j^{(k)} = \frac{1}{\zeta} * \frac{1}{e^{\kappa_j \mu_a}}

    where :math:`\zeta` is the continuous input precision (in real space).


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
    # Value coupling updates
    # ----------------------
    if edges[node_idx].value_children is not None:
        precision_weigthed_prediction_error = 0.0
        for value_child_idx, value_coupling in zip(
            edges[node_idx].value_children,  # type: ignore
            attributes[node_idx]["value_coupling_children"],
        ):
            expected_precision_children = attributes[value_child_idx][
                "expected_precision_children"
            ]
            precision_weigthed_prediction_error += (
                expected_precision_children * value_coupling
            )

        # Compute the new posterior precision
        posterior_precision = (
            attributes[node_idx]["expected_precision"]
            + precision_weigthed_prediction_error
        )

    # Volatility coupling updates
    # ---------------------------
    if edges[node_idx].volatility_children is not None:
        precision_weigthed_prediction_error = 0.0
        for volatility_child_idx, volatility_coupling in zip(
            edges[node_idx].volatility_children,  # type: ignore
            attributes[node_idx]["volatility_coupling_children"],
        ):
            # volatility weigthed precision for the volatility child (γ)
            volatility_weigthed_precision = (
                attributes[volatility_child_idx]["expected_precision"]
                * attributes[volatility_child_idx]["temp"]["predicted_volatility"]
            )

            # retrieve the volatility prediction error
            volatility_prediction_error = attributes[volatility_child_idx]["temp"][
                "volatility_prediction_error"
            ]

            # sum over all volatility children
            precision_weigthed_prediction_error += (
                0.5 * (volatility_coupling * volatility_weigthed_precision) ** 2
                + (volatility_coupling * volatility_weigthed_precision) ** 2
                * volatility_prediction_error
                - 0.5
                * volatility_coupling**2
                * volatility_weigthed_precision
                * volatility_prediction_error
            )

        # Compute the new posterior precision
        posterior_precision = (
            attributes[node_idx]["expected_precision"]
            + precision_weigthed_prediction_error
        )

    # ensure the new precision is greater than 0
    posterior_precision = jnp.where(
        posterior_precision <= 0, jnp.nan, posterior_precision
    )

    return posterior_precision


@partial(jit, static_argnames=("edges", "node_idx"))
def update_continuous_node(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Prediction-error step for the value and volatility parents of a continuous node.

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
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # update the posterior mean and precision
    posterior_precision = posterior_update_precision_continuous_node(
        attributes, edges, node_idx
    )
    posterior_mean = posterior_update_mean_continuous_node(
        attributes, edges, node_idx, node_precision=attributes[node_idx]["precision"]
    )

    attributes[node_idx]["precision"] = posterior_precision
    attributes[node_idx]["mean"] = posterior_mean

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def ehgf_update_continuous_node(
    attributes: Dict, time_step: float, node_idx: int, edges: Edges, **args
) -> Dict:
    """Perform the eHGF PE step for value and volatility parents of a continuous node.

    This update step uses a different order for the mean and precision as compared to
    the standard HGF, respectively:
    1. Update volatility parent(s).
    2. Update value parent(s).

    If a value/volatility parent has multiple children, all the children will update
    the parent together, therefor this function should only be called once per group
    of child nodes. The method :py:meth:`pyhgf.model.HGF.get_update_sequence`
    ensures that this function is only called once all the children have been
    updated.

    Then returns the structure of the new parameters.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the node that needs to be updated. After continuous updates, the
        parameters of value and volatility parents (if any) will be different.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    update_continuous_input_parents, update_binary_input_parents

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
        attributes, edges, node_idx
    )
    attributes[node_idx]["precision"] = posterior_precision

    return attributes
