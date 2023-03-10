# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import Dict, Optional, Tuple

from jax import jit


def loop_inputs(
    parameters_structure: Dict,
    data: Tuple,
    update_sequence: Tuple,
    node_structure: Tuple,
) -> Tuple[Dict, Dict]:
    """Update the node structure after observing one new data point.

    One time step updating node structure and returning the new node structure with
    time, value and surprise stored in the input node.

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
    data :
        The current array element. Scan will iterate over a n x 2 Array with time steps
        and values (i.e. the input time series).
    update_sequence :
        The sequence of updates that will be applied to the node structure.
    node_structure :
        The node structure with the update sequence.

    """
    # Extract the current iteration variables (value and time)
    value, time_step = data

    # Fit the model with the current time and value variables, given model structure
    new_parameters_structure = apply_sequence(
        value=value,
        time_step=time_step,
        parameters_structure=parameters_structure,
        node_structure=node_structure,
        update_sequence=update_sequence,
    )

    return (
        new_parameters_structure,
        new_parameters_structure,
    )  # ("carryover", "accumulated")


@partial(jit, static_argnums=(3, 4))
def apply_sequence(
    value: Optional[float],
    time_step: Optional[float],
    parameters_structure: Dict,
    node_structure: Dict,
    update_sequence: Tuple,
) -> Dict:
    """Apply a predefinded update sequence to a node structure."""
    for sequence in update_sequence:
        node_idx, update_fn = sequence
        parameters_structure = update_fn(
            parameters_structure=parameters_structure,
            time_step=time_step,
            node_idx=node_idx,
            node_structure=node_structure,
            value=value,
        )
    return parameters_structure
