# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import scan, switch
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.networks import (
    beliefs_propagation,
    fill_categorical_state_node,
    get_update_sequence,
    to_pandas,
)
from pyhgf.plots import plot_correlations, plot_network, plot_nodes, plot_trajectories
from pyhgf.response import first_level_binary_surprise, first_level_gaussian_surprise
from pyhgf.typing import Edges, Indexes, InputIndexes, UpdateSequence


class HGF(object):
    r"""The two-level and three-level Hierarchical Gaussian Filters (HGF).

    This class uses pre-made node structures that correspond to the most widely used
    HGF. The inputs can be continuous or binary.

    Attributes
    ----------
    attributes :
        The attributes of the probabilistic nodes.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as the node
        number. For each node, the index lists the value/volatility parents/children.
    input_nodes_idx :
        Indexes of the input nodes with input types
        :py:class:`pyhgf.typing.InputIndexes`. The default input node is `0` if the
        network only has one input node.
    model_type :
        The model implemented (can be `"continuous"`, `"binary"` or `"custom"`).
    n_levels :
        The number of hierarchies in the model, including the input vector. It cannot be
        less than 2.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    update_sequence :
        The sequence of update functions that are applied during the belief propagation
        step.
    update_type :
        The type of update to perform for volatility coupling. Can be `"eHGF"`
        (defaults) or `"standard"`. The eHGF update step was proposed as an alternative
        to the original definition in that it starts by updating the mean and then the
        precision of the parent node, which generally reduces the errors associated with
        impossible parameter space and improves sampling.
    .. note::
        The parameter structure also incorporates the value and volatility coupling
        strength with children and parents (i.e. `"value_coupling_parents"`,
        `"value_coupling_children"`, `"volatility_coupling_parents"`,
        `"volatility_coupling_children"`).
    scan_fn :
        The function that is passed to :py:func:`jax.lax.scan`. This is a pre-
        parametrized version of :py:func:`pyhgf.networks.beliefs_propagation`.
    verbose : bool
        Verbosity level.

    """

    def __init__(
        self,
        n_levels: Optional[int] = 2,
        model_type: str = "continuous",
        update_type: str = "eHGF",
        initial_mean: Dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
        initial_precision: Dict = {
            "1": 1.0,
            "2": 1.0,
            "3": 1.0,
        },
        continuous_precision: Union[float, np.ndarray, ArrayLike] = 1e4,
        tonic_volatility: Dict = {
            "1": -3.0,
            "2": -3.0,
            "3": -3.0,
        },
        volatility_coupling: Dict = {"1": 1.0, "2": 1.0},
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
        binary_precision: Union[float, np.ndarray, ArrayLike] = jnp.inf,
        tonic_drift: Dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
        verbose: bool = True,
    ):
        r"""Parameterization of the HGF model.

        Parameters
        ----------
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterwards.
            Defaults to `2` for a 2-level HGF.
        model_type : str
            The model type to use (can be `"continuous"` or `"binary"`).
        update_type:
            The type of volatility update to perform. It can be `"eHGF"` (default) or
            `"standard"`. The eHGF update step tends to be more robust and produce fewer
            invalid space errors and is therefore recommended by default.
        initial_mean :
            A dictionary containing the initial values for the initial mean at
            different levels of the hierarchy. Defaults set to `0.0`.
        initial_precision :
            A dictionary containing the initial values for the initial precision at
            different levels of the hierarchy. Defaults set to `1.0`.
        continuous_precision :
            The expected precision of the continuous input node. Default to `1e4`. Only
            relevant if `model_type="continuous"`.
        tonic_volatility :
            A dictionary containing the initial values for the tonic volatility
            at different levels of the hierarchy. This represents the tonic
            part of the variance (the part that is not affected by the parent node).
            Defaults are set to `-3.0`.
        volatility_coupling :
            A dictionary containing the initial values for the volatility coupling
            at different levels of the hierarchy. This represents the phasic part of
            the variance (the part that is affected by the parent nodes) and will
            define the strength of the connection between the node and the parent
            node. Defaults set to `1.0`.
        eta0 :
            The first categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        eta1 :
            The second categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        binary_precision :
            The precision of the binary input node. Default to `jnp.inf`. Only relevant
            if `model_type="binary"`.
        tonic_drift :
            A dictionary containing the initial values for the tonic drift
            at different levels of the hierarchy. This represents the drift of the
            random walk. Defaults set all entries to `0.0` (no drift).
        verbose :
            The verbosity of the methods for model creation and fitting. Defaults to
            `True`.

        """
        self.model_type = model_type
        self.update_type = update_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.edges: Edges = ()
        self.node_trajectories: Dict
        self.attributes: Dict = {}
        self.update_sequence: Optional[UpdateSequence] = None
        self.scan_fn: Optional[Callable] = None

        if model_type not in ["continuous", "binary"]:
            if self.verbose:
                print("Initializing a network with custom node structure.")
        else:
            if self.verbose:
                print(
                    (
                        f"Creating a {self.model_type} Hierarchical Gaussian Filter "
                        f"with {self.n_levels} levels."
                    )
                )
            if model_type == "continuous":
                # Input
                self.add_nodes(
                    kind="continuous-input",
                    node_parameters={"continuous_precision": continuous_precision},
                )

                # X-1
                self.add_nodes(
                    value_children=([0], [1.0]),
                    node_parameters={
                        "mean": initial_mean["1"],
                        "precision": initial_precision["1"],
                        "tonic_volatility": tonic_volatility["1"],
                        "tonic_drift": tonic_drift["1"],
                    },
                )

                # X-2
                self.add_nodes(
                    volatility_children=([1], [volatility_coupling["1"]]),
                    node_parameters={
                        "mean": initial_mean["2"],
                        "precision": initial_precision["2"],
                        "tonic_volatility": tonic_volatility["2"],
                        "tonic_drift": tonic_drift["2"],
                    },
                )

            elif model_type == "binary":
                # Input
                self.add_nodes(
                    kind="binary-input",
                    node_parameters={
                        "eta0": eta0,
                        "eta1": eta1,
                        "binary_precision": binary_precision,
                    },
                )

                # X -1
                self.add_nodes(
                    kind="binary-state",
                    value_children=([0], [1.0]),
                    node_parameters={
                        "mean": initial_mean["1"],
                        "precision": initial_precision["1"],
                    },
                )

                # X -2
                self.add_nodes(
                    kind="continuous-state",
                    value_children=([1], [1.0]),
                    node_parameters={
                        "mean": initial_mean["2"],
                        "precision": initial_precision["2"],
                        "tonic_volatility": tonic_volatility["2"],
                        "tonic_drift": tonic_drift["2"],
                    },
                )

            #########
            # x - 3 #
            #########
            if self.n_levels == 3:
                self.add_nodes(
                    volatility_children=([2], [volatility_coupling["2"]]),
                    node_parameters={
                        "mean": initial_mean["3"],
                        "precision": initial_precision["3"],
                        "tonic_volatility": tonic_volatility["3"],
                        "tonic_drift": tonic_drift["3"],
                    },
                )

            # initialize the model so it is ready to receive new observations
            self.create_belief_propagation_fn()

    def create_belief_propagation_fn(self, overwrite: bool = True) -> "HGF":
        """Create the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`.

        Parameters
        ----------
        overwrite :
            If `True` (default), create a new belief propagation function and ignore
            preexisting values. Otherwise, do not create a new function if the attribute
            `scan_fn` is already defined.

        """
        # create the update sequence if it does not already exist
        if self.update_sequence is None:
            self.set_update_sequence()
            if self.verbose:
                print("... Create the update sequence from the network structure.")

        # create the belief propagation function
        # this function is laer used by scan to loop over observations
        if self.scan_fn is None:
            self.scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                edges=self.edges,
                input_nodes_idx=self.input_nodes_idx.idx,
            )
            if self.verbose:
                print("... Create the belief propagation function.")
        else:
            if overwrite:
                self.scan_fn = Partial(
                    beliefs_propagation,
                    update_sequence=self.update_sequence,
                    edges=self.edges,
                    input_nodes_idx=self.input_nodes_idx.idx,
                )
                if self.verbose:
                    print("... Create the belief propagation function (overwrite).")
            else:
                if self.verbose:
                    print("... The belief propagation function is already defined.")

        return self

    def cache_belief_propagation_fn(self) -> "HGF":
        """Blank call to the belief propagation function.

        .. note:
           This step is called by default when using py:meth:`input_data`. It can
           sometimes be convenient to call this step independently to chache the JITed
           function before fitting the model.

        """
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()

        # blanck call to cache the JIT-ed functions
        _ = scan(
            self.scan_fn,
            self.attributes,
            (
                jnp.ones((1, len(self.input_nodes_idx.idx))),
                jnp.ones((1, 1)),
                jnp.ones((1, 1)),
            ),
        )
        if self.verbose:
            print("... Cache the belief propagation function.")

        return self

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
    ) -> "HGF":
        """Add new observations.

        Parameters
        ----------
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
            prediction errors, they are updated by keeping the same mean by decreasing
            the precision as a function of time to reflect the evolution of the
            underlying Gaussian Random Walk.
        .. warning::
            Missing inputs are missing observations from the agent's perspective and
            should not be used to handle missing data points that were observed (e.g.
            missing in the event log, or rejected trials).

        """
        if self.scan_fn is None:
            self = self.create_belief_propagation_fn()
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones((len(input_data), 1))  # time steps vector
        else:
            time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        # is it observation or missing inputs
        if observed is None:
            observed = np.ones(input_data.shape, dtype=int)

        # this is where the model loops over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        _, node_trajectories = scan(
            self.scan_fn, self.attributes, (input_data, time_steps, observed)
        )

        # trajectories of the network attributes a each time point
        self.node_trajectories = node_trajectories

        return self

    def input_custom_sequence(
        self,
        update_branches: Tuple[UpdateSequence],
        branches_idx: np.array,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
        observed: Optional[np.ndarray] = None,
    ):
        """Add new observations with custom update sequences.

        This method should be used when the update sequence should be adapted to the
        input data. (e.g. in the case of missing/null observations that should not
        trigger node update).

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

        """
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones(len(input_data))  # time steps vector

        # concatenate data and time
        if time_steps is None:
            time_steps = np.ones((len(input_data), 1))  # time steps vector
        else:
            time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        # is it observation or missing inputs
        if observed is None:
            observed = np.ones(input_data.shape, dtype=int)

        # create the update functions that will be scanned
        branches_fn = [
            Partial(
                beliefs_propagation,
                update_sequence=seq,
                edges=self.edges,
                input_nodes_idx=self.input_nodes_idx.idx,
            )
            for seq in update_branches
        ]

        # create the function that will be scanned
        def switching_propagation(attributes, scan_input):
            data, idx = scan_input
            return switch(idx, branches_fn, attributes, data)

        # wrap the inputs
        scan_input = (input_data, time_steps, observed), branches_idx

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(switching_propagation, self.attributes, scan_input)

        # the node structure at each value updates
        self.node_trajectories = node_trajectories

        # because some of the input nodes might not have been updated, here we manually
        # insert the input data to the input node (without triggering updates)
        for idx, inp in zip(self.input_nodes_idx.idx, range(input_data.shape[1])):
            self.node_trajectories[idx]["value"] = input_data[inp]

        return self

    def plot_nodes(self, node_idxs: Union[int, List[int]], **kwargs):
        """Plot the node(s) beliefs trajectories."""
        return plot_nodes(hgf=self, node_idxs=node_idxs, **kwargs)

    def plot_trajectories(self, **kwargs):
        """Plot the parameters trajectories."""
        return plot_trajectories(hgf=self, **kwargs)

    def plot_correlations(self):
        """Plot the heatmap of cross-trajectories correlation."""
        return plot_correlations(hgf=self)

    def plot_network(self):
        """Visualization of node network using GraphViz."""
        return plot_network(hgf=self)

    def surprise(
        self,
        response_function: Optional[Callable] = None,
        response_function_inputs: Tuple = (),
        response_function_parameters: Optional[
            Union[np.ndarray, ArrayLike, float]
        ] = None,
    ):
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
        if response_function is None:
            response_function = (
                first_level_gaussian_surprise
                if self.model_type == "continuous"
                else first_level_binary_surprise
            )

        return response_function(
            hgf=self,
            response_function_inputs=response_function_inputs,
            response_function_parameters=response_function_parameters,
        )

    def to_pandas(self) -> pd.DataFrame:
        """Export the nodes trajectories and surprise as a Pandas data frame.

        Returns
        -------
        structure_df :
            Pandas data frame with the time series of sufficient statistics and
            the surprise of each node in the structure.

        """
        return to_pandas(self)

    def add_nodes(
        self,
        kind: str = "continuous-state",
        n_nodes: int = 1,
        node_parameters: Dict = {},
        value_children: Optional[Union[List, Tuple, int]] = None,
        value_parents: Optional[Union[List, Tuple, int]] = None,
        volatility_children: Optional[Union[List, Tuple, int]] = None,
        volatility_parents: Optional[Union[List, Tuple, int]] = None,
    ):
        """Add new input/state node(s) to the neural network.

        Parameters
        ----------
        kind :
            The kind of node to create. If `"continuous-state"` (default), the node will
            be a regular state node that can have value and/or volatility
            parents/children. The hierarchical dependencies are specified using the
            corresponding parameters below. If `"binary-state"`, the node should be the
            value parent of a binary input. To create an input node, three types of
            inputs are supported:
            - `continuous-input`: receive a continuous observation as input.
            - `binary-input` receive a single boolean as observation. The parameters
            provided to the binary input node contain: 1. `binary_precision`, the binary
            input precision, which defaults to `jnp.inf`. 2. `eta0`, the lower bound of
            the binary process, which defaults to `0.0`. 3. `eta1`, the higher bound of
            the binary process, which defaults to `1.0`.
            - `categorical-input` receive a boolean array as observation. The parameters
            provided to the categorical input node contain: 1. `n_categories`, the
            number of categories implied by the categorical state.

        .. note::
           When using a categorical state node, the `binary_parameters` can be used to
           parametrize the implied collection of binray HGFs.

        .. note:
           When using `categorical-input`, the implied `n` binary HGFs are automatically
           created with a shared volatility parent at the third level, resulting in a
           network with `3n + 2` nodes in total.

        n_nodes :
            The number of nodes to create (defaults to `1`).
        node_parameters :
            Dictionary of parameters. The default values are automatically inferred
            from the node type. Different values can be provided by passing them in the
            dictionary, which will overwrite the defaults.
        value_children:
            Indexes to the node's value children. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        value_parents:
            Indexes to the node's value parents. The index can be passed as an integer
            or a list of integers, in case of multiple children. The coupling strength
            can be controlled by passing a tuple, where the first item is the list of
            indexes, and the second item is the list of coupling strengths.
        volatility_children:
            Indexes to the node's volatility children. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.
        volatility_parents:
            Indexes to the node's volatility parents. The index can be passed as an
            integer or a list of integers, in case of multiple children. The coupling
            strength can be controlled by passing a tuple, where the first item is the
            list of indexes, and the second item is the list of coupling strengths.

        """
        # extract the node coupling indexes and coupling strengths
        couplings = []
        for indexes in [
            value_children,
            value_parents,
            volatility_children,
            volatility_parents,
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

        # create the default parameters set according to the node type
        if kind == "continuous-state":
            default_parameters = {
                "mean": 0.0,
                "expected_mean": 0.0,
                "precision": 1.0,
                "expected_precision": 1.0,
                "volatility_coupling_children": couplings[2][1],
                "volatility_coupling_parents": couplings[3][1],
                "value_coupling_children": couplings[0][1],
                "value_coupling_parents": couplings[1][1],
                "tonic_volatility": -4.0,
                "tonic_drift": 0.0,
                "autoregressive_coefficient": 0.0,
                "autoregressive_intercept": 0.0,
                "observed": 1,
                "temp": {
                    "effective_precision": 0.0,
                    "value_prediction_error": 0.0,
                    "volatility_prediction_error": 0.0,
                    "expected_precision_children": 0.0,
                },
            }
        elif kind == "binary-state":
            default_parameters = {
                "mean": 0.0,
                "expected_mean": 0.0,
                "precision": 1.0,
                "expected_precision": 1.0,
                "volatility_coupling_children": couplings[2][1],
                "volatility_coupling_parents": couplings[3][1],
                "value_coupling_children": couplings[0][1],
                "value_coupling_parents": couplings[1][1],
                "tonic_volatility": 0.0,
                "tonic_drift": 0.0,
                "autoregressive_coefficient": 0.0,
                "autoregressive_intercept": 0.0,
                "observed": 1,
                "binary_expected_precision": 0.0,
                "temp": {
                    "effective_precision": 0.0,
                    "value_prediction_error": 0.0,
                    "volatility_prediction_error": 0.0,
                    "expected_precision_children": 0.0,
                },
            }
        elif kind == "continuous-input":
            default_parameters = {
                "volatility_coupling_parents": None,
                "value_coupling_parents": None,
                "input_noise": 1e4,
                "expected_precision": 1e4,
                "time_step": 0.0,
                "value": 0.0,
                "surprise": 0.0,
                "observed": 0,
                "temp": {
                    "effective_precision": 1.0,  # should be fixed to 1 for input nodes
                    "value_prediction_error": 0.0,
                    "volatility_prediction_error": 0.0,
                },
            }
        elif kind == "binary-input":
            default_parameters = {
                "expected_precision": jnp.inf,
                "eta0": 0.0,
                "eta1": 1.0,
                "time_step": 0.0,
                "value": 0.0,
                "observed": 0,
                "surprise": 0.0,
            }
        elif kind == "categorical-input":
            if "n_categories" in node_parameters:
                n_categories = node_parameters["n_categories"]
            else:
                n_categories = 4
            binary_parameters = {
                "eta0": 0.0,
                "eta1": 1.0,
                "binary_precision": jnp.inf,
                "n_categories": n_categories,
                "precision_1": 1.0,
                "precision_2": 1.0,
                "precision_3": 1.0,
                "mean_1": 0.0,
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
                "time_step": jnp.nan,
                "alpha": jnp.ones(n_categories),
                "pe": jnp.zeros(n_categories),
                "xi": jnp.array([1.0 / n_categories] * n_categories),
                "mean": jnp.array([1.0 / n_categories] * n_categories),
                "value": jnp.zeros(n_categories),
                "binary_parameters": binary_parameters,
            }

        # update the defaults using the provided parameters
        default_parameters.update(node_parameters)
        node_parameters = default_parameters

        if "input" in kind:
            input_type = kind.split("-")[0]
        else:
            input_type = None

        # convert the structure to a list to modify it
        edges_as_list: List[Indexes] = list(self.edges)

        for _ in range(n_nodes):
            node_idx = len(self.attributes)  # the index of the new node

            # add a new edge
            edges_as_list.append(
                Indexes(
                    couplings[1][0], couplings[3][0], couplings[0][0], couplings[2][0]
                )
            )

            if node_idx == 0:
                # this is the first node, create the node structure
                self.attributes = {node_idx: node_parameters}
                if input_type is not None:
                    self.input_nodes_idx = InputIndexes((node_idx,), (input_type,))
            else:
                # update the node structure
                self.attributes[node_idx] = node_parameters

                if input_type is not None:
                    # add information about the new input node in the indexes
                    new_idx = self.input_nodes_idx.idx
                    new_idx += (node_idx,)
                    new_kind = self.input_nodes_idx.kind
                    new_kind += (input_type,)
                    self.input_nodes_idx = InputIndexes(new_idx, new_kind)

            # update the existing edge structure so it links to the new node as well
            for coupling, edge_type in zip(
                couplings,
                [
                    "value_children",
                    "value_parents",
                    "volatility_children",
                    "volatility_parents",
                ],
            ):
                if coupling[0] is not None:
                    coupling_idxs, coupling_strengths = coupling
                    for idx, coupling_strength in zip(
                        coupling_idxs, coupling_strengths  # type: ignore
                    ):
                        # unpack this node's edges
                        (
                            value_parents,
                            volatility_parents,
                            value_children,
                            volatility_children,
                        ) = edges_as_list[idx]

                        # update the parents/children's edges depending on the coupling
                        if edge_type == "value_parents":
                            if value_children is None:
                                value_children = (node_idx,)
                                self.attributes[idx]["value_coupling_children"] = (
                                    coupling_strength,
                                )
                            else:
                                value_children = value_children + (node_idx,)
                                self.attributes[idx]["value_coupling_children"] += (
                                    coupling_strength,
                                )
                        elif edge_type == "volatility_parents":
                            if volatility_children is None:
                                volatility_children = (node_idx,)
                                self.attributes[idx]["volatility_coupling_children"] = (
                                    coupling_strength,
                                )
                            else:
                                volatility_children = volatility_children + (node_idx,)
                                self.attributes[idx][
                                    "volatility_coupling_children"
                                ] += (coupling_strength,)
                        elif edge_type == "value_children":
                            if value_parents is None:
                                value_parents = (node_idx,)
                                self.attributes[idx]["value_coupling_parents"] = (
                                    coupling_strength,
                                )
                            else:
                                value_parents = value_parents + (node_idx,)
                                self.attributes[idx]["value_coupling_parents"] += (
                                    coupling_strength,
                                )
                        elif edge_type == "volatility_children":
                            if volatility_parents is None:
                                volatility_parents = (node_idx,)
                                self.attributes[idx]["volatility_coupling_parents"] = (
                                    coupling_strength,
                                )
                            else:
                                volatility_parents = volatility_parents + (node_idx,)
                                self.attributes[idx]["volatility_coupling_parents"] += (
                                    coupling_strength,
                                )

                        # save the updated edges back
                        edges_as_list[idx] = Indexes(
                            value_parents,
                            volatility_parents,
                            value_children,
                            volatility_children,
                        )

        # convert the list back to a tuple
        self.edges = tuple(edges_as_list)

        # if we are creating a categorical state or state-transition node
        # we have to generate the implied binary network(s) here
        if kind == "categorical-input":
            self = fill_categorical_state_node(
                self,
                node_idx=node_idx,
                binary_input_idxs=node_parameters["binary_idxs"],  # type: ignore
                binary_parameters=binary_parameters,
            )

        return self

    def set_update_sequence(self) -> "HGF":
        """Generate an update sequence from the network's structure.

        See :py:func:`pyhgf.networks.get_update_sequence` for more details.

        """
        self.update_sequence = tuple(
            get_update_sequence(
                hgf=self,
            )
        )

        return self
