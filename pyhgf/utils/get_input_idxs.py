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


def get_input_idxs(edges: Edges) -> Tuple[int, ...]:
    """List all possible default inputs nodes.

    An input node is a state node without any child.

    Parameters
    ----------
    edges :
        The edges of the probabilistic network as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the number of
        nodes. For each node, the index list value/volatility - parents/children.

    """
    return tuple(
        [
            i
            for i in range(len(edges))
            if (
                (edges[i].value_children is None)
                & (edges[i].volatility_children is None)
            )
        ]
    )
