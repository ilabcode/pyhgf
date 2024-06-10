# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.math import gaussian_surprise
from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_volatility_prediction_error(
    attributes: Dict, edges: Edges, node_idx: int
) -> Dict:
    r"""Store noise prediction error from an input node.

    Input nodes can have noise parents, and therefore should compute the equivalent
    of a volatility prediction error (VOPE): a noise prediction error (NOPE). The
    volatility parent will use the NOPE value the same way that it uses VOPE values.
    Note that the effective precision :math:`\gamma_j^{(k)}` is fixed to `1` so this
    equivalence applies.

    The noise prediction error :math:`\epsilon_j^{(k)}` of an input node :math:`j` is
    given by:

    .. math::

        \epsilon_j^{(k)} = \frac{\hat{\pi}_j^{(k)}}{\pi_{vapa}^{(k)}} +
            \hat{\pi}_j^{(k)} \left( u^{(k)} - \mu_{vapa}^{(k)} \right)^2 - 1

    Note that, because we are working with continuous input nodes,
    :math:`\epsilon_j^{(k)}` is not a function of the value prediction error but uses
    the posterior of the value parent(s).

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
        Pointer to the input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    See Also
    --------
    continuous_input_value_prediction_error, continuous_input_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    expected_precision_input = attributes[node_idx]["expected_precision"]
    value_parent_idx = edges[node_idx].value_parents[0]  # type: ignore

    # compute the noise prediction error for this input node
    noise_prediction_error = (
        (expected_precision_input / attributes[value_parent_idx]["precision"])
        + expected_precision_input
        * attributes[node_idx]["temp"]["value_prediction_error"] ** 2
        - 1
    )

    # use the noise prediction error (NOPE) as VOPE
    attributes[node_idx]["temp"]["volatility_prediction_error"] = noise_prediction_error

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_value_prediction_error(
    attributes: Dict,
    edges: Edges,
    time_step: float,
    node_idx: int,
) -> Dict:
    r"""Store value prediction error and expected precision from an input node.

    The value prediction error :math:`\delta_j^{(k)}` of an input node :math:`j` at time
    :math:`k` is given by:

    .. math::

        \delta_j^{(k)} = u^{(k)} - \hat{\mu_j}^{(k)}

    The expected precision :math:`\hat{\pi}_j^{(k)}` of an input node is the sum of the
    tonic and phasic volatility, which is given by:

    .. math::

        \hat{\pi}_j^{(k)} = \frac{1}{\zeta} * \frac{1}{e^{\kappa_j \mu_a}}

    where :math:`\zeta` is the input precision (in real space).

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    time_step :
        The time interval between the previous time point and the current time point.
    node_idx :
        Pointer to the input node.

    Returns
    -------
    attributes :
        The attributes of the probabilistic nodes.

    Notes
    -----
    This update step is similar to the one used for the state node, except that it uses
    the observed value instead of the mean of the child node, and the expected mean of
    the parent node instead of the expected mean of the child node.

    See Also
    --------
    continuous_input_volatility_prediction_error, continuous_input_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # retireve the index of the value parent (assuming a unique value parent)
    value_parent_idx = edges[node_idx].value_parents[0]  # type: ignore

    # the prediction error is computed using the expected mean from the value parent
    value_prediction_error = (
        attributes[node_idx]["values"] - attributes[value_parent_idx]["expected_mean"]
    )

    # expected precision from the input node
    expected_precision = attributes[node_idx]["input_precision"]

    # influence of a volatility parent on the input node
    volatility_parents_idxs = edges[node_idx].volatility_parents  # type:ignore
    if volatility_parents_idxs is not None:
        for volatility_parents_idx, volatility_coupling in zip(
            volatility_parents_idxs,
            attributes[node_idx]["volatility_coupling_parents"],
        ):
            expected_precision *= 1 / (
                time_step
                * jnp.exp(
                    attributes[volatility_parents_idx]["expected_mean"]
                    * volatility_coupling
                )
            )

    # store in the current node for later use in the update step
    attributes[node_idx]["temp"]["value_prediction_error"] = value_prediction_error
    attributes[node_idx]["expected_precision"] = expected_precision

    # compute the Gaussian surprise for the input node
    attributes[node_idx]["surprise"] = gaussian_surprise(
        x=attributes[node_idx]["values"],
        expected_mean=attributes[value_parent_idx]["expected_mean"],
        expected_precision=expected_precision,
    )
    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_prediction_error(
    attributes: Dict,
    time_step: float,
    node_idx: int,
    edges: Edges,
    value: float,
    observed: bool,
) -> Dict:
    """Store prediction errors in an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    node_idx :
        Pointer to the input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value and volatility parents and
        children.
    value :
        The new observed value.
    observed :
        Whether value was observed or not.

    Returns
    -------
    attributes :
        The updated attributes of the probabilistic nodes.

    See Also
    --------
    continuous_input_value_prediction_error, continuous_input_value_prediction_error

    References
    ----------
    .. [1] Weber, L. A., Waade, P. T., Legrand, N., Møller, A. H., Stephan, K. E., &
       Mathys, C. (2023). The generalized Hierarchical Gaussian Filter (Version 1).
       arXiv. https://doi.org/10.48550/ARXIV.2305.10937

    """
    # store value and time step in the node's parameters
    attributes[node_idx]["values"] = value
    attributes[node_idx]["observed"] = observed
    attributes[node_idx]["time_step"] = time_step

    if (edges[node_idx].value_parents is not None) or (
        edges[node_idx].volatility_parents is not None
    ):
        # Store value prediction errors
        # -----------------------------
        attributes = continuous_input_value_prediction_error(
            attributes=attributes, edges=edges, node_idx=node_idx, time_step=time_step
        )

        # Store volatility prediction errors
        # ----------------------------------
        attributes = continuous_input_volatility_prediction_error(
            attributes=attributes,
            edges=edges,
            node_idx=node_idx,
        )

    return attributes
