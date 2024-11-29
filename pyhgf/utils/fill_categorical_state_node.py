# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import jit
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.math import binary_surprise, gaussian_surprise
from pyhgf.typing import AdjacencyLists, Attributes, Edges, Sequence, UpdateSequence
from pyhgf.updates.observation import set_observation
from pyhgf.updates.posterior.categorical import categorical_state_update
from pyhgf.updates.posterior.continuous import (
    continuous_node_posterior_update,
    continuous_node_posterior_update_ehgf,
)
from pyhgf.updates.prediction.binary import binary_state_node_prediction
from pyhgf.updates.prediction.continuous import continuous_node_prediction
from pyhgf.updates.prediction.dirichlet import dirichlet_node_prediction
from pyhgf.updates.prediction_error.binary import binary_state_node_prediction_error
from pyhgf.updates.prediction_error.categorical import (
    categorical_state_prediction_error,
)
from pyhgf.updates.prediction_error.continuous import continuous_node_prediction_error
from pyhgf.updates.prediction_error.dirichlet import dirichlet_node_prediction_error
from pyhgf.updates.prediction_error.exponential import (
    prediction_error_update_exponential_family,
)

if TYPE_CHECKING:
    from pyhgf.model import Network


def fill_categorical_state_node(
    network: "Network",
    node_idx: int,
    binary_states_idxs: List[int],
    binary_parameters: Dict,
) -> "Network":
    """Generate a binary network implied by categorical state(-transition) nodes.

    Parameters
    ----------
    network :
        Instance of a Network.
    node_idx :
        Index to the categorical state node.
    binary_states_idxs :
        The indexes of the binary state nodes.
    binary_parameters :
        Parameters for the set of implied binary HGFs.

    Returns
    -------
    hgf :
        The updated instance of the HGF model.

    """
    # add the binary states - one for each category
    network.add_nodes(
        kind="binary-state",
        n_nodes=len(binary_states_idxs),
        node_parameters={
            "mean": binary_parameters["mean_1"],
            "precision": binary_parameters["precision_1"],
        },
    )

    # add the value coupling between the categorical and binary states
    edges_as_list: List[AdjacencyLists] = list(network.edges)
    edges_as_list[node_idx] = AdjacencyLists(
        5, tuple(binary_states_idxs), None, None, None, (None,)
    )
    for binary_idx in binary_states_idxs:
        edges_as_list[binary_idx] = AdjacencyLists(
            1, None, None, (node_idx,), None, (None,)
        )
    network.edges = tuple(edges_as_list)

    # add continuous state parent nodes
    n_nodes = len(network.edges)
    for i in range(binary_parameters["n_categories"]):
        network.add_nodes(
            value_children=i + n_nodes - binary_parameters["n_categories"],
            node_parameters={
                "mean": binary_parameters["mean_2"],
                "precision": binary_parameters["precision_2"],
                "tonic_volatility": binary_parameters["tonic_volatility_2"],
            },
        )

    # add the higher level volatility parents
    # as a shared parents between the second level nodes
    network.add_nodes(
        volatility_children=[
            idx + binary_parameters["n_categories"] for idx in binary_states_idxs
        ],
        node_parameters={
            "mean": binary_parameters["mean_3"],
            "precision": binary_parameters["precision_3"],
            "tonic_volatility": binary_parameters["tonic_volatility_3"],
        },
    )

    return network
