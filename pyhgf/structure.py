# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Optional, Tuple

from jax import jit

from pyhgf.typing import NodeStructure


def loop_inputs(
    update_sequence: Tuple, node_structure: NodeStructure, data: Tuple
) -> Tuple[Dict, Dict]:
    """Update the node structure after observing one new data point.

    One time step updating node structure and returning the new node structure with
    time, value and surprise stored in the input node.

    Parameters
    ----------
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    node_structure :
        The node structure with the update sequence.
    data :
        The current array element. Scan will iterate over a n x 2 DeviceArray with time
        and values (i.e. the input time series).

    """
    # Extract the current iteration variables (value and time)
    value, time_step = data

    # Fit the model with the current time and value variables, given model structure
    new_node_structure = apply_sequence(
        node_structure=node_structure,
        update_sequence=update_sequence,
        value=value,
        time_step=time_step,
    )

    return new_node_structure, new_node_structure  # ("carryover", "accumulated")


@partial(jit, static_argnums=(1))
def apply_sequence(
    node_structure: NodeStructure,
    update_sequence: Tuple,
    time_step: Optional[float] = None,
    value: Optional[float] = None,
):
    """Apply a predefinded update sequence to a node structure."""
    for sequence in update_sequence:
        node_idx, value_parents_idx, volatility_parents_idx, update_fn = sequence
        node_structure = update_fn(
            node_structure=node_structure,
            node_idx=node_idx,
            value_parents_idx=value_parents_idx,
            volatility_parents_idx=volatility_parents_idx,
            time_step=time_step,
            value=value,
        )
    return node_structure
