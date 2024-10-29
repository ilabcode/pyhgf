# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import scan, switch
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.plots import plot_correlations, plot_network, plot_nodes, plot_trajectories
from pyhgf.typing import (
    AdjacencyLists,
    Attributes,
    Edges,
    NetworkParameters,
    UpdateSequence,
)
from pyhgf.utils import (
    add_edges,
    beliefs_propagation,
    fill_categorical_state_node,
    get_input_idxs,
    get_update_sequence,
    to_pandas,
)


class Network:
    """A predictive coding neural network.

    This is the core class to define and manipulate neural networks, that consists in
    1. attributes, 2. structure and 3. update sequences.

    Attributes
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.AdjacencyLists`. The tuple has the same length as the
        node number. For each node, the index lists the value/volatility
        parents/children.
    inputs :
        Information on the input nodes.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    update_sequence :
        The sequence of update functions that are applied during the belief propagation
        step.
    scan_fn :
        The function that is passed to :py:func:`jax.lax.scan`. This is a pre-
        parametrized version of :py:func:`pyhgf.networks.beliefs_propagation`.

    """

    def __init__(self) -> None:
        """Initialize an empty neural network."""
        self.edges: Edges = ()
        self.node_trajectories: Dict = {}
        self.attributes: Attributes = {-1: {"time_step": 0.0}}
        self.update_sequence: Optional[UpdateSequence] = None
        self.scan_fn: Optional[Callable] = None

    @property
    def input_idxs(self):
        """Idexes of state nodes that can observe new data points by default."""
        input_idxs = get_input_idxs(self.edges)

        # set the autoconnection strength and tonic volatility to 0
        for idx in input_idxs:
            if self.edges[idx].node_type == 2:
                self.attributes[idx]["autoconnection_strength"] = 0.0
                self.attributes[idx]["tonic_volatility"] = 0.0
        return input_idxs

    @input_idxs.setter
    def input_idxs(self, value):
        self.input_idxs = value

    def create_belief_propagation_fn(
        self, overwrite: bool = True, update_type: str = "eHGF"
    ) -> "Network":
        """Create the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`.

        Parameters
        ----------
        overwrite :
            If `True` (default), create a new belief propagation function and ignore
            preexisting values. Otherwise, do not create a new function if the attribute
            `scan_fn` is already defined.
        update_type :
            The type of update to perform for volatility coupling. Can be `"eHGF"`
            (defaults) or `"standard"`. The eHGF update step was proposed as an
            alternative to the original definition in that it starts by updating the
            mean and then the precision of the parent node, which generally reduces the
            errors associated with impossible parameter space and improves sampling.

        """
        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.update_sequence = get_update_sequence(
                network=self, update_type=update_type
            )

        # create the belief propagation function
        # this function is used by scan to loop over observations
        if (self.scan_fn is None) or overwrite:
            self.scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                edges=self.edges,
                input_idxs=self.input_idxs,
            )

        return self

    def input_data(
        self,
        input_data: Union[np.ndarray, tuple],
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[Union[np.ndarray, tuple]] = None,
        input_idxs: Optional[Tuple[int]] = None,
    ):
        """Add new observations.

        Parameters
        ----------
        input_data :
            2d array of new observations (time x features).
        time_steps :
            Time steps vector (optional). If `None`, this will default to
            `np.ones(len(input_data))`.
        observed :
            A time * input node boolean array masking `input_data`. In case of missing
            inputs, (i.e. `observed` is `0`), the input node will have value and
            volatility set to `0.0`. If the parent(s) of this input receive prediction
            error from other children, they simply ignore this one. If they are not
            receiving other prediction errors, they are updated by keeping the same
            mean by decreasing the precision as a function of time to reflect the
            evolution of the underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).
        input_idxs :
            Indexes on the state nodes receiving observations.

        """
        # set the input nodes indexes
        if input_idxs is not None:
            self.input_idxs = input_idxs

        # belief propagation function
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        # input_data should be a tuple of n by time_steps arrays
        if not isinstance(input_data, tuple):
            if observed is None:
                observed = np.ones(input_data.shape, dtype=int)
            if input_data.ndim == 1:

                # Interleave observations and masks
                input_data = (input_data, observed)
            else:
                observed = jnp.hsplit(observed, input_data.shape[1])
                observations = jnp.hsplit(input_data, input_data.shape[1])

                # Interleave observations and masks
                input_data = tuple(
                    [
                        item.flatten()
                        for pair in zip(observations, observed)
                        for item in pair
                    ]
                )

        # time steps vector
        if time_steps is None:
            time_steps = np.ones(input_data[0].shape[0])

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        last_attributes, node_trajectories = scan(
            self.scan_fn, self.attributes, (*input_data, time_steps)
        )

        # belief trajectories
        self.node_trajectories = node_trajectories
        self.last_attributes = last_attributes

        return self

    def input_custom_sequence(
        self,
        update_branches: Tuple[UpdateSequence],
        branches_idx: np.array,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
        input_idxs: Optional[Tuple[int]] = None,
    ):
        """Add new observations with custom update sequences.

        This method should be used when the update sequence is function of the input
        data. (e.g. in the case of missing/null observations that should not trigger
        node update).

        .. note::
           When the dynamic adaptation of the update sequence is not required, it is
           recommended to use :py:meth:`pyhgf.model.HGF.input_data` instead as this
           might result in performance improvement.

        Parameters
        ----------
        update_branches :
            A tuple of UpdateSequence listing the possible update sequences.
        branches_idx :
            The branches indexes (integers). Should have the same length as the input
            data.
        input_data :
            2d array of new observations (time x features).
        time_steps :
            Time vector (optional). If `None`, the time vector will default to
            `np.ones(len(input_data))`. This vector is automatically transformed
            into a time steps vector.
        observed :
            A 2d boolean array masking `input_data`. In case of missing inputs, (i.e.
            `observed` is `0`), the input node will have value and volatility set to
            `0.0`. If the parent(s) of this input receive prediction error from other
            children, they simply ignore this one. If they are not receiving other
            prediction errors, they are updated by keeping the same mean be decreasing
            the precision as a function of time to reflect the evolution of the
            underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).
        input_idxs :
            Indexes on the state nodes receiving observations.

        """
        # set the input nodes indexes
        if input_idxs is not None:
            self.input_idxs = input_idxs

        # input_data should be a tuple of n by time_steps arrays
        if not isinstance(input_data, tuple):
            if observed is None:
                observed = np.ones(input_data.shape, dtype=int)
            if input_data.ndim == 1:

                # Interleave observations and masks
                input_data = (input_data, observed)
            else:
                observed = jnp.hsplit(observed, input_data.shape[1])
                observations = jnp.hsplit(input_data, input_data.shape[1])

                # Interleave observations and masks
                input_data = tuple(
                    [item for pair in zip(observations, observed) for item in pair]
                )

        # time steps vector
        if time_steps is None:
            time_steps = np.ones(input_data[0].shape[0])

        # create the update functions that will be scanned
        branches_fn = [
            Partial(
                beliefs_propagation,
                update_sequence=seq,
                edges=self.edges,
                input_idxs=self.input_idxs,
            )
            for seq in update_branches
        ]

        # create the function that will be scanned
        def switching_propagation(attributes, scan_input):
            (*data, idx) = scan_input
            return switch(idx, branches_fn, attributes, data)

        # wrap the inputs
        scan_input = (*input_data, time_steps, branches_idx)

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(switching_propagation, self.attributes, scan_input)

        # the node structure at each value updates
        self.node_trajectories = node_trajectories

        return self

    def get_network(self) -> NetworkParameters:
        """Return the attributes, edges and update sequence defining the network."""
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        assert self.update_sequence is not None

        return self.attributes, self.edges, self.update_sequence

    def add_nodes(
        self,
        kind: str = "continuous-state",
        n_nodes: int = 1,
        node_parameters: Dict = {},
        value_children: Optional[Union[List, Tuple, int]] = None,
        value_parents: Optional[Union[List, Tuple, int]] = None,
        volatility_children: Optional[Union[List, Tuple, int]] = None,
        volatility_parents: Optional[Union[List, Tuple, int]] = None,
        coupling_fn: Tuple[Optional[Callable], ...] = (None,),
        **additional_parameters,
    ):
        """Add new input/state node(s) to the neural network.

        Parameters
        ----------
        kind :
            The kind of node to create. If `"continuous-state"` (default), the node will
            be a regular state node that can have value and/or volatility
            parents/children. If `"binary-state"`, the node should be the
            value parent of a binary input. State nodes filtering distribution from the
            exponential family can be created using `"exponential-state"`.

        .. note::
            When using a categorical state node, the `binary_parameters` can be used to
            parametrize the implied collection of binary HGFs.

        .. note:
            When using `categorical-state`, the implied `n` binary HGFs are
            automatically created with a shared volatility parent at the third level,
            resulting in a network with `3n + 2` nodes in total.

        n_nodes :
            The number of nodes to create (defaults to `1`).
        node_parameters :
            Dictionary of parameters. The default values are automatically inferred
            from the node type. Different values can be provided by passing them in the
            dictionary, which will overwrite the defaults.
        value_children :
            Indexes to the node's value children. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        value_parents :
            Indexes to the node's value parents. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        volatility_children :
            Indexes to the node's volatility children. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.
        volatility_parents :
            Indexes to the node's volatility parents. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.
        coupling_fn :
            Coupling function(s) between the current node and its value children.
            It has to be provided as a tuple. If multiple value children are specified,
            the coupling functions must be stated in the same order of the children.
            Note: if a node has multiple parents nodes with different coupling
            functions, a coupling function should be indicated for all the parent nodes.
            If no coupling function is stated, the relationship between nodes is assumed
            linear.
        **kwargs :
            Additional keyword parameters will be passed and overwrite the node
            attributes.

        """
        if kind not in [
            "DP-state",
            "exponential-state",
            "categorical-state",
            "continuous-state",
            "binary-state",
            "generic-state",
        ]:
            raise ValueError(
                (
                    "Invalid node type. Should be one of the following: "
                    "'DP-state', 'continuous-state', 'binary-state', "
                    "'exponential-state', 'generic-state' or 'categorical-state'"
                )
            )

        # assess children number
        # this is required to ensure the coupling functions match
        children_number = 1
        if value_children is None:
            children_number = 0
        elif isinstance(value_children, int):
            children_number = 1
        elif isinstance(value_children, list):
            children_number = len(value_children)

        # transform coupling parameter into tuple of indexes and strenghts
        couplings = []
        for indexes in [
            value_parents,
            volatility_parents,
            value_children,
            volatility_children,
        ]:
            if indexes is not None:
                if isinstance(indexes, int):
                    coupling_idxs = tuple([indexes])
                    coupling_strengths = tuple([1.0])
                elif isinstance(indexes, list):
                    coupling_idxs = tuple(indexes)
                    coupling_strengths = tuple([1.0] * len(coupling_idxs))
                elif isinstance(indexes, tuple):
                    coupling_idxs = tuple(indexes[0])
                    coupling_strengths = tuple(indexes[1])
            else:
                coupling_idxs, coupling_strengths = None, None
            couplings.append((coupling_idxs, coupling_strengths))
        value_parents, volatility_parents, value_children, volatility_children = (
            couplings
        )

        # create the default parameters set according to the node type
        if kind == "continuous-state":
            default_parameters = {
                "mean": 0.0,
                "expected_mean": 0.0,
                "precision": 1.0,
                "expected_precision": 1.0,
                "volatility_coupling_children": volatility_children[1],
                "volatility_coupling_parents": volatility_parents[1],
                "value_coupling_children": value_children[1],
                "value_coupling_parents": value_parents[1],
                "tonic_volatility": -4.0,
                "tonic_drift": 0.0,
                "autoconnection_strength": 1.0,
                "observed": 1,
                "temp": {
                    "effective_precision": 0.0,
                    "value_prediction_error": 0.0,
                    "volatility_prediction_error": 0.0,
                },
            }
        elif kind == "binary-state":
            default_parameters = {
                "observed": 1,
                "mean": 0,
                "expected_mean": 0.5,
                "precision": 1.0,
                "expected_precision": 1.0,
                "value_coupling_parents": value_parents[1],
                "temp": {
                    "value_prediction_error": 0.0,
                },
            }
        elif kind == "generic-state":
            default_parameters = {
                "mean": 0.0,
                "observed": 1,
            }
        elif "exponential-state" in kind:
            default_parameters = {
                "nus": 3.0,
                "xis": jnp.array([0.0, 1.0]),
                "mean": 0.0,
                "observed": 1,
            }
        elif kind == "categorical-state":
            if "n_categories" in node_parameters:
                n_categories = node_parameters["n_categories"]
            elif "n_categories" in additional_parameters:
                n_categories = additional_parameters["n_categories"]
            else:
                n_categories = 4
            binary_parameters = {
                "n_categories": n_categories,
                "precision_1": 1.0,
                "precision_2": 1.0,
                "precision_3": 1.0,
                "mean_1": 1 / n_categories,
                "mean_2": -jnp.log(n_categories - 1),
                "mean_3": 0.0,
                "tonic_volatility_2": -4.0,
                "tonic_volatility_3": -4.0,
            }
            binary_idxs: List[int] = [
                1 + i + len(self.edges) for i in range(n_categories)
            ]
            default_parameters = {
                "binary_idxs": binary_idxs,  # type: ignore
                "n_categories": n_categories,
                "surprise": 0.0,
                "kl_divergence": 0.0,
                "alpha": jnp.ones(n_categories),
                "observed": jnp.ones(n_categories, dtype=int),
                "mean": jnp.array([1.0 / n_categories] * n_categories),
                "binary_parameters": binary_parameters,
            }
        elif kind == "DP-state":

            if "batch_size" in additional_parameters.keys():
                batch_size = additional_parameters["batch_size"]
            elif "batch_size" in node_parameters.keys():
                batch_size = node_parameters["batch_size"]
            else:
                batch_size = 10

            default_parameters = {
                "batch_size": batch_size,  # number of branches available in the network
                "n": jnp.zeros(batch_size),  # number of observation in each cluster
                "n_total": 0,  # the total number of observations in the node
                "alpha": 1.0,  # concentration parameter for the implied Dirichlet dist.
                "expected_means": jnp.zeros(batch_size),
                "expected_sigmas": jnp.ones(batch_size),
                "sensory_precision": 1.0,
                "activated": jnp.zeros(batch_size),
                "value_coupling_children": (1.0,),
                "mean": 0.0,
                "n_active_cluster": 0,
            }

        # Update the default node parameters using keywords args and dictonary
        if bool(additional_parameters):
            # ensure that all passed values are valid keys
            invalid_keys = [
                key
                for key in additional_parameters.keys()
                if key not in default_parameters.keys()
            ]

            if invalid_keys:
                raise ValueError(
                    (
                        "Some parameter(s) passed as keyword arguments were not found "
                        f"in the default key list for this node (i.e. {invalid_keys})."
                        " If you want to create a new key in the node attributes, "
                        "please use the node_parameters argument instead."
                    )
                )

            # if keyword parameters were provided, update the default_parameters dict
            default_parameters.update(additional_parameters)

        # update the defaults using the dict parameters
        default_parameters.update(node_parameters)
        node_parameters = default_parameters

        # define the type of node that is created
        if kind == "generic-state":
            node_type = 0
        elif kind == "binary-state":
            node_type = 1
        elif kind == "continuous-state":
            node_type = 2
        elif kind == "exponential-state":
            node_type = 3
        elif kind == "DP-state":
            node_type = 4
        elif kind == "categorical-state":
            node_type = 5

        for _ in range(n_nodes):
            # convert the structure to a list to modify it
            edges_as_list: List = list(self.edges)

            node_idx = len(self.edges)  # the index of the new node

            # for mutiple value children, set a default tuple with corresponding length
            if children_number != len(coupling_fn):
                if coupling_fn == (None,):
                    coupling_fn = children_number * coupling_fn
                else:
                    raise ValueError(
                        "The number of coupling fn and value children do not match"
                    )

            # add a new edge
            edges_as_list.append(
                AdjacencyLists(
                    node_type, None, None, None, None, coupling_fn=coupling_fn
                )
            )

            # convert the list back to a tuple
            self.edges = tuple(edges_as_list)

            # update the node structure
            self.attributes[node_idx] = deepcopy(node_parameters)

            # Update the edges of the parents and children accordingly
            # --------------------------------------------------------
            if value_parents[0] is not None:
                self.add_edges(
                    kind="value",
                    parent_idxs=value_parents[0],
                    children_idxs=node_idx,
                    coupling_strengths=value_parents[1],  # type: ignore
                )
            if value_children[0] is not None:
                self.add_edges(
                    kind="value",
                    parent_idxs=node_idx,
                    children_idxs=value_children[0],
                    coupling_strengths=value_children[1],  # type: ignore
                    coupling_fn=coupling_fn,
                )
            if volatility_children[0] is not None:
                self.add_edges(
                    kind="volatility",
                    parent_idxs=node_idx,
                    children_idxs=volatility_children[0],
                    coupling_strengths=volatility_children[1],  # type: ignore
                )
            if volatility_parents[0] is not None:
                self.add_edges(
                    kind="volatility",
                    parent_idxs=volatility_parents[0],
                    children_idxs=node_idx,
                    coupling_strengths=volatility_parents[1],  # type: ignore
                )

        if kind == "categorical-state":
            # if we are creating a categorical state or state-transition node
            # we have to generate the implied binary network(s) here
            self = fill_categorical_state_node(
                self,
                node_idx=node_idx,
                binary_states_idxs=node_parameters["binary_idxs"],  # type: ignore
                binary_parameters=binary_parameters,
            )

        return self

    def plot_nodes(self, node_idxs: Union[int, List[int]], **kwargs):
        """Plot the node(s) beliefs trajectories."""
        return plot_nodes(network=self, node_idxs=node_idxs, **kwargs)

    def plot_trajectories(self, **kwargs):
        """Plot the parameters trajectories."""
        return plot_trajectories(network=self, **kwargs)

    def plot_correlations(self):
        """Plot the heatmap of cross-trajectories correlation."""
        return plot_correlations(network=self)

    def plot_network(self):
        """Visualization of node network using GraphViz."""
        return plot_network(network=self)

    def to_pandas(self) -> pd.DataFrame:
        """Export the nodes trajectories and surprise as a Pandas data frame.

        Returns
        -------
        structure_df :
            Pandas data frame with the time series of sufficient statistics and
            the surprise of each node in the structure.

        """
        return to_pandas(self)

    def surprise(
        self,
        response_function: Callable,
        response_function_inputs: Tuple = (),
        response_function_parameters: Optional[
            Union[np.ndarray, ArrayLike, float]
        ] = None,
    ) -> float:
        """Surprise of the model conditioned by the response function.

        The surprise (negative log probability) depends on the input data, the model
        parameters, the response function, its inputs and its additional parameters
        (optional).

        Parameters
        ----------
        response_function :
            The response function to use to compute the model surprise. If `None`
            (default), return the sum of Gaussian surprise if `model_type=="continuous"`
            or the sum of the binary surprise if `model_type=="binary"`.
        response_function_inputs :
            A list of tuples with the same length as the number of models. Each tuple
            contains additional data and parameters that can be accessible to the
            response functions.
        response_function_parameters :
            A list of additional parameters that will be passed to the response
            function. This can include values over which inferece is performed in a
            PyMC model (e.g. the inverse temperature of a binary softmax).

        Returns
        -------
        surprise :
            The model's surprise given the input data and the response function.

        """
        return response_function(
            hgf=self,
            response_function_inputs=response_function_inputs,
            response_function_parameters=response_function_parameters,
        )
        return self

    def add_edges(
        self,
        kind="value",
        parent_idxs=Union[int, List[int]],
        children_idxs=Union[int, List[int]],
        coupling_strengths: Union[float, List[float], Tuple[float]] = 1.0,
        coupling_fn: Tuple[Optional[Callable], ...] = (None,),
    ) -> "Network":
        """Add a value or volatility coupling link between a set of nodes.

        Parameters
        ----------
        kind :
            The kind of coupling, can be `"value"` or `"volatility"`.
        parent_idxs :
            The index(es) of the parent node(s).
        children_idxs :
            The index(es) of the children node(s).
        coupling_strengths :
            The coupling strength betwen the parents and children.
        coupling_fn :
            Coupling function(s) between the current node and its value children.
            It has to be provided as a tuple. If multiple value children are specified,
            the coupling functions must be stated in the same order of the children.
            Note: if a node has multiple parents nodes with different coupling
            functions, a coupling function should be indicated for all the parent nodes.
            If no coupling function is stated, the relationship between nodes is assumed
            linear.

        """
        attributes, edges = add_edges(
            attributes=self.attributes,
            edges=self.edges,
            kind=kind,
            parent_idxs=parent_idxs,
            children_idxs=children_idxs,
            coupling_strengths=coupling_strengths,
            coupling_fn=coupling_fn,
        )

        self.attributes = attributes
        self.edges = edges

        return self
