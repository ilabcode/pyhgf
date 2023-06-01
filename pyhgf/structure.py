# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike


def loop_inputs(
    parameters_structure: Dict,
    data: Tuple,
    update_sequence: Tuple,
    node_structure: Tuple,
    input_nodes_idx: ArrayLike = [0],
) -> Tuple[Dict, Dict]:
    """Update the node structure after observing new data point(s).

    One time step updating node structure and returning the new parameter structure.

    Parameters
    ----------
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        `"psis"` is the value coupling strength. It should have same length than the
        volatility parents' indexes. `"kappas"` is the volatility coupling strength.
        It should have same length than the volatility parents' indexes.
    data :
        A tuple of (data, time_steps). `data` is a n x j+1 Array (j the number of
        inputs), the last dimension is time.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    node_structure :
        The node structure with the update sequence.
    input_nodes_idx :
        List of input indexes. Defaults to `[0]`.

    """
    # Extract the current iteration variables (value and time)
    values = data[:-1]
    time_step = data[-1]

    # Fit the model with the current time and value variables, given model structure
    new_parameters_structure = apply_sequence(
        values=values,
        time_step=time_step,
        parameters_structure=parameters_structure,
        node_structure=node_structure,
        update_sequence=update_sequence,
        input_nodes_idx=input_nodes_idx,
    )

    return (
        new_parameters_structure,
        new_parameters_structure,
    )  # ("carryover", "accumulated")


@partial(jit, static_argnums=(3, 4))
def apply_sequence(
    values: ArrayLike,
    time_step: Optional[float],
    parameters_structure: Dict,
    node_structure: Dict,
    update_sequence: Tuple,
    input_nodes_idx: ArrayLike,
) -> Dict:
    """Apply an update sequence to a node structure.

    Parameters
    ----------
    values :
        The new observation(s). This can be a single value or a vector of observation
        with length matching `input_nodes_idx`. `input_nodes_idx` is used to index the
        ith value in the vector to the ith input node, so the order matters.
    time_step :
        The time interval between the new observation and the previous one.
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        `"psis"` is the value coupling strength. It should have same length than the
        volatility parents' indexes. `"kappas"` is the volatility coupling strength.
        It should have same length than the volatility parents' indexes.
    node_structure :
        The node structure with the update sequence.
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    input_nodes_idx :
        List of input indexes. Defaults to `[0]`.

    """
    for sequence in update_sequence:
        node_idx, update_fn = sequence

        # if we are updating an input node, select the value that should be passed
        value = jnp.where(
            jnp.isin(node_idx, input_nodes_idx),
            jnp.sum(jnp.equal(input_nodes_idx, node_idx) * values),
            jnp.nan,
        )

        parameters_structure = update_fn(
            parameters_structure=parameters_structure,
            time_step=time_step,
            node_idx=node_idx,
            node_structure=node_structure,
            value=value,
        )
    return parameters_structure
