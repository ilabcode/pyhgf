# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict

import jax.numpy as jnp
from jax import jit

from pyhgf.typing import Edges


@partial(jit, static_argnames=("edges", "volatility_parent_idx"))
def continuous_input_volatility_prediction_error(
    attributes: Dict, edges: Edges, time_step: float, input_idx: int
) -> Dict:
    r"""Store volatility prediction error from an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    time_step :
        The interval between the previous time point and the current time point.
    input_idx :
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
    expected_precision_input = attributes[input_idx]["expected_precision"]
    value_parent_idx = edges[input_idx].value_parents[0]  # type: ignore

    # compute the noise prediction error for this input node
    noise_prediction_error = (
        (expected_precision_input / attributes[value_parent_idx]["precision"])
        + expected_precision_input
        * attributes[input_idx]["value_prediction_error"] ** 2
        - 1
    )

    # use the noise prediction error (NOPE) as VOPE
    attributes[input_idx]["temp"][
        "volatility_prediction_error"
    ] = noise_prediction_error

    return attributes


@partial(jit, static_argnames=("edges", "value_parent_idx"))
def continuous_input_value_prediction_error(
    attributes: Dict,
    edges: Edges,
    input_idx: int,
) -> Dict:
    r"""Store value prediction error and expected precision from an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    input_idx :
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
    value_parent_idx = edges[input_idx].value_parents[0]  # type: ignore

    # the prediction error is computed using the expected mean from the value parent
    value_prediction_error = (
        attributes[input_idx]["value"] - attributes[value_parent_idx]["expected_mean"]
    )

    # expected precision from the input node
    expected_precision_child = attributes[input_idx]["expected_precision"]

    # influence of a volatility parent on the input node
    if edges[input_idx].volatility_parents is not None:
        volatility_parent = edges[input_idx].volatility_parents[0]  # type:ignore
        volatility_coupling = attributes[volatility_parent][
            "volatility_coupling_children"
        ][0]
        expected_precision_child *= 1 / jnp.exp(
            attributes[volatility_parent]["expected_mean"] * volatility_coupling
        )

    # store in the current node for later use in the update step
    attributes[input_idx]["temp"]["value_prediction_error"] = value_prediction_error
    attributes[input_idx]["temp"][
        "expected_precision_children"
    ] = expected_precision_child

    return attributes


@partial(jit, static_argnames=("edges", "node_idx"))
def continuous_input_prediction_error(
    attributes: Dict,
    time_step: float,
    input_idx: int,
    edges: Edges,
    value: float,
) -> Dict:
    """Store prediction errors in an input node.

    Parameters
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    time_step :
        The interval between the previous time point and the current time point.
    input_idx :
        Pointer to the input node.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    value :
        The new observed value.

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
    attributes[input_idx]["value"] = value
    attributes[input_idx]["time_step"] = time_step

    # Store value prediction errors
    # -----------------------------
    if edges[input_idx].value_parents is not None:
        attributes = continuous_input_value_prediction_error(
            attributes=attributes, edges=edges, input_idx=input_idx
        )

    # Store volatility prediction errors
    # ----------------------------------
    if edges[input_idx].volatility_parents is not None:
        attributes = continuous_input_volatility_prediction_error(
            attributes=attributes, edges=edges, input_idx=input_idx
        )

    return attributes
