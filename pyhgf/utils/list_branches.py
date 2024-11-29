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


def list_branches(node_idxs: List, edges: Tuple, branch_list: List = []) -> List:
    """Return the branch of a network from a given set of root nodes.

    This function searches recursively and lists the parents above a given node. If all
    the children of a given parent are on the exclusion list, this parent is also
    excluded.

    Parameters
    ----------
    node_idxs :
        A list of node indexes. The nodes can be input nodes or any other node in the
        network.
    edges :
        The nodes structure.
    branch_list :
        The list of nodes that are already excluded (i.e )

    Returns
    -------
    branch_list :
        The list of node indexes that belong to the branch.

    """
    for idx in node_idxs:
        # add this node to the exclusion list
        branch_list.append(idx)
        all_parents = np.array(
            [
                i
                for i in [
                    edges[idx].value_parents,
                    edges[idx].volatility_parents,
                ]
                if i is not None
            ]
        ).flatten()
        for parent_idx in all_parents:
            # list the children for this node
            all_children = np.array(
                [
                    i
                    for i in [
                        edges[parent_idx].value_children,
                        edges[parent_idx].volatility_children,
                    ]
                    if i is not None
                ]
            ).flatten()
            # if this parent has only excluded children, add it to the exclusion list
            if np.all([i in branch_list for i in all_children]):
                branch_list = list_branches(
                    [parent_idx], edges, branch_list=branch_list
                )

    return branch_list

