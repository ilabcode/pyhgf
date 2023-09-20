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

    This class uses pre-made nodes structures that correspond to the most widely used
    HGF. The inputs can be continuous or binary.

    Attributes
    ----------
    input_nodes_idx :
        Indexes of the input nodes with input types
        :py:class:`pyhgf.typing.InputIndexes`. The default input node is `0` if the
        network only has one input node.
    model_type :
        The model implemented (can be `"continuous"`, `"binary"` or `"custom"`).
    n_levels :
        The number of hierarchies in the model, including the input vector. Cannot be
        less than 2.
    edges :
        The edges of the probabilistic nodes as a tuple of
        :py:class:`pyhgf.typing.Indexes`. The tuple has the same length as node number.
        For each node, the index list value and volatility parents and children.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    attributes :
        The attributes of the probabilistic nodes.
    .. note::
        The parameter structure also incorporate the value and volatility coupling
        strenght with children and parents (i.e. `"psis_parents"`, `"psis_children"`,
        `"kappas_parents"`, `"kappas_children"`).
    verbose : bool
        Verbosity level.

    """

    def __init__(
        self,
        n_levels: Optional[int] = 2,
        model_type: str = "continuous",
        initial_mu: Dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
        initial_pi: Dict = {
            "1": 1.0,
            "2": 1.0,
            "3": 1.0,
        },
        continuous_precision: Union[float, np.ndarray, ArrayLike] = 1e4,
        omega: Dict = {
            "1": -3.0,
            "2": -3.0,
            "3": -3.0,
        },
        kappas: Dict = {"1": 1.0, "2": 1.0},
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
        binary_precision: Union[float, np.ndarray, ArrayLike] = jnp.inf,
        rho: Dict = {
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
        },
        verbose: bool = True,
    ):
        r"""Parameterization of the HGF model.

        Parameters
        ----------
        eta0 :
            The first categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        eta1 :
            The second categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        initial_mu :
            A dictionary containing the initial values for the :math:`\\mu` parameter at
            different levels of the hierarchy. Defaults set to `0.0` at all levels.
        initial_pi :
            A dictionary containing the initial values for the :math:`\\pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults are set to `1.0` at all levels.
        kappas :
            A dictionary containing the initial values for the :math:`\\kappa` parameter
            at different levels of the hierarchy. Kappa represents the phasic part of
            the variance (the part that is affected by the parent nodes) and will
            define the strength of the connection between the node and the parent
            node. Often fixed to `1`. Defaults are set to `1` at all levels.
        model_type : str
            The model type to use (can be `"continuous"` or `"binary"`).
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Defaults to `2` for a 2-level HGF.
        omega :
            A dictionary containing the initial values for the :math:`\\omega` parameter
            at different levels of the hierarchy. :math:`\\omega` represents the tonic
            part of the variance (the part that is not affected by the parent node).
            Defaults are set to `-3.0` at all levels. This parameter is only required
            for Gaussian Random Walk nodes.
        continuous_precision :
            The expected precision of the continuous input node. Default to `1e4`. Only
            relevant if `model_type="continuous"`.
        binary_precision :
            The precision of the binary input node. Default to `jnp.inf`. Only relevant
            if `model_type="binary"`.
        rho :
            A dictionary containing the initial values for the :math:`\\rho` parameter
            at different levels of the hierarchy. Rho represents the drift of the random
            walk. This parameter is only required for Gaussian Random Walk nodes.
            Defaults set all entries to `0.0` according to the number of required
            levels.
        verbose :
            The verbosity of the methods for model creation and fitting. Defaults to
            `True`.

        """
        self.model_type = model_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.edges: Edges
        self.node_trajectories: Dict
        self.attributes: Dict
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
            #########
            # Input #
            #########
            if model_type == "continuous":
                self.add_input_node(
                    kind="continuous",
                    continuous_parameters={
                        "continuous_precision": continuous_precision
                    },
                )
            elif model_type == "binary":
                self.add_input_node(
                    kind="binary",
                    binary_parameters={
                        "eta0": eta0,
                        "eta1": eta1,
                        "binary_precision": binary_precision,
                    },
                )

            #########
            # x - 1 #
            #########
            self.add_value_parent(
                children_idxs=[0],
                value_coupling=1.0,
                mu=initial_mu["1"],
                pi=initial_pi["1"],
                omega=omega["1"] if self.model_type != "binary" else jnp.nan,
                rho=rho["1"] if self.model_type != "binary" else jnp.nan,
            )

            #########
            # x - 2 #
            #########
            if model_type == "continuous":
                self.add_volatility_parent(
                    children_idxs=[1],
                    volatility_coupling=kappas["1"],
                    mu=initial_mu["2"],
                    pi=initial_pi["2"],
                    omega=omega["2"],
                    rho=rho["2"],
                )
            elif model_type == "binary":
                self.add_value_parent(
                    children_idxs=[1],
                    value_coupling=1.0,
                    mu=initial_mu["2"],
                    pi=initial_pi["2"],
                    omega=omega["2"],
                    rho=rho["2"],
                )

            #########
            # x - 3 #
            #########
            if self.n_levels == 3:
                self.add_volatility_parent(
                    children_idxs=[2],
                    volatility_coupling=kappas["2"],
                    mu=initial_mu["3"],
                    pi=initial_pi["3"],
                    omega=omega["3"],
                    rho=rho["3"],
                )

            # initialize the model so it is ready to receive new observations
            self.init()

    def init(self) -> "HGF":
        """Initialize the update functions.

        This step should be called after creating the network to compile the update
        sequence and before providing any input data.

        """
        # create the update sequence automatically if not provided
        if self.update_sequence is None:
            self.set_update_sequence()
            if self.verbose:
                print("... Create the update sequence from the network structure.")

        # create the belief propagation function that will be scaned
        if self.scan_fn is None:
            self.scan_fn = Partial(
                beliefs_propagation,
                update_sequence=self.update_sequence,
                edges=self.edges,
                input_nodes_idx=self.input_nodes_idx.idx,
            )
            if self.verbose:
                print("... Create the belief propagation function.")

        # blanck call to cache the JIT-ed functions
        _ = scan(
            self.scan_fn,
            self.attributes,
            jnp.array([jnp.ones(len(self.input_nodes_idx.idx) + 1)]),
        )
        if self.verbose:
            print("... Cache the belief propagation function.")

        return self

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
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

        """
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones(len(input_data))  # time steps vector

        # concatenate data and time
        time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        data = jnp.concatenate((input_data, time_steps), dtype=float, axis=1)

        # this is where the model loop over the whole input time series
        # at each time point, the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        _, node_trajectories = scan(self.scan_fn, self.attributes, data)

        # trajectories of the network attributes a each time point
        self.node_trajectories = node_trajectories

        return self

    def input_custom_sequence(
        self,
        update_branches: Tuple[UpdateSequence],
        branches_idx: np.array,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
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

        """
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time_steps is None:
            time_steps = np.ones(len(input_data))  # time steps vector

        # concatenate data and time
        time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]
        data = jnp.concatenate((input_data, time_steps), dtype=float, axis=1)

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
        scan_input = data, branches_idx

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(switching_propagation, self.attributes, scan_input)

        # the node structure at each value update
        self.node_trajectories = node_trajectories

        # because some of the input node might not have been updated, here we manually
        # insert the input data to the input node (without triggering updates)
        for idx, inp in zip(self.input_nodes_idx.idx, range(input_data.shape[1])):
            self.node_trajectories[idx]["value"] = input_data[:, inp]

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
        response_function_parameters: Tuple = (),
    ):
        """Surprise (negative log probability) under new observations.

        The surprise depends on the input data, the model parameters, the response
        function and its additional parameters(optional).

        Parameters
        ----------
        response_function :
            The response function to use to compute the model surprise. If `None`
            (default), return the sum of Gaussian surprise if `model_type=="continuous"`
            or the sum of the binary surprise if `model_type=="binary"`.
        response_function_parameters :
            Additional parameters to the response function (optional).

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

    def add_input_node(
        self,
        kind: str,
        input_idxs: Union[List, int] = 0,
        continuous_parameters: Dict = {"continuous_precision": 1e4},
        binary_parameters: Dict = {
            "eta0": 0.0,
            "eta1": 1.0,
            "binary_precision": jnp.inf,
        },
        categorical_parameters: Dict = {"n_categories": 4},
        additional_parameters: Optional[Dict] = None,
    ):
        """Create an input node.

        Three types of input nodes are supported:

        - `continuous`: receive a continuous observation as input. The parameter
        `continuous_precision` is required.
        - `binary` receive a single boolean as observation. The parameters
        `binary_precision`, `eta0` and `eta1` are required.
        - `categorical` receive a boolean array as observation. The parameters are
        `n_categories` required.

        .. note:
           When using `categorical`, the implied `n` binary HGFs are automatically
           created with a shared volatility parent at the third level, resulting in a
           network with `3n + 2` nodes in total.

        Parameters
        ----------
        kind :
            The kind of input that should be created (can be `"continuous"`, `"binary"`
            or `"categorical"`).
        input_idxs :
            The index of the new input (defaults to `0`).
        continuous_parameters :
            Parameters provided to the continuous input node. Contains:
            - `continuous_precision` (the precision of the observations), which
              defaults to `1e4`.
        binary_parameters :
            Parameters provided to the binary input node. Contains:
            - `binary_precision`, the binary input precision, which defaults to
            `jnp.inf`.
            - `eta0`, the lower bound of the binary process. Defaults to `0.0`.
            - `eta1`, the higher bound of the binary process. Defaults to `1.0`.
        categorical_parameters :
            Parameters provided to the categorical input node. Contains:
            - `n_categories`, the number of categories implied by the categorical state
            node(s).
            .. note::
               When using a categorical state node, the `binary_parameters` can be used
               to parametrize the implied collection of binray HGFs.
        additional_parameters :
            Add more custom parameters to the input node.

        """
        if not isinstance(input_idxs, list):
            input_idxs = [input_idxs]

        if kind == "continuous":
            input_node_parameters = {
                "kappas_parents": None,
                "pihat": continuous_parameters["continuous_precision"],
                "time_step": jnp.nan,
                "value": jnp.nan,
            }
        elif kind == "binary":
            input_node_parameters = {
                "pihat": binary_parameters["binary_precision"],
                "eta0": binary_parameters["eta0"],
                "eta1": binary_parameters["eta1"],
                "surprise": jnp.nan,
                "time_step": jnp.nan,
                "value": jnp.nan,
            }
        elif kind == "categorical":
            input_node_parameters = {
                "binary_surprise": 0.0,
                "kl_divergence": 0.0,
                "time_step": jnp.nan,
                "alpha": jnp.ones(categorical_parameters["n_categories"]),
                "pe": jnp.zeros(categorical_parameters["n_categories"]),
                "xi": jnp.array(
                    [1.0 / categorical_parameters["n_categories"]]
                    * categorical_parameters["n_categories"]
                ),
                "mu": jnp.array(
                    [1.0 / categorical_parameters["n_categories"]]
                    * categorical_parameters["n_categories"]
                ),
                "value": jnp.zeros(categorical_parameters["n_categories"]),
            }

        # add more parameters (optional)
        if additional_parameters is not None:
            input_node_parameters = {**input_node_parameters, **additional_parameters}

        for input_idx in input_idxs:
            if input_idx == 0:
                # this is the first node, create the node structure
                self.attributes = {input_idx: input_node_parameters}
                self.edges = (Indexes(None, None, None, None),)
                self.input_nodes_idx = InputIndexes((input_idx,), (kind,))
            else:
                # update the node structure
                self.attributes[input_idx] = input_node_parameters
                self.edges += (Indexes(None, None, None, None),)

                # add information about the new input node in the indexes
                new_idx = self.input_nodes_idx.idx
                new_idx += (input_idx,)
                new_kind = self.input_nodes_idx.kind
                new_kind += (kind,)
                self.input_nodes_idx = InputIndexes(new_idx, new_kind)

            # if we are creating a categorical state or state-transition node
            # we have to generate the implied binary network(s) here
            if kind == "categorical":
                implied_binary_parameters = {
                    "eta0": 0.0,
                    "eta1": 1.0,
                    "binary_precision": jnp.inf,
                    "binary_idxs": [
                        i + len(self.edges)
                        for i in range(categorical_parameters["n_categories"])
                    ],
                    "n_categories": categorical_parameters["n_categories"],
                    "pi_1": 1.0,
                    "pi_2": 1.0,
                    "pi_3": 1.0,
                    "mu_1": 0.0,
                    "mu_2": -jnp.log(categorical_parameters["n_categories"] - 1),
                    "mu_3": 0.0,
                    "omega_2": -4.0,
                    "omega_3": -4.0,
                }
                implied_binary_parameters.update(binary_parameters)

                self = fill_categorical_state_node(
                    self,
                    node_idx=input_idx,
                    implied_binary_parameters=implied_binary_parameters,
                )

        return self

    def add_value_parent(
        self,
        children_idxs: Union[List, int],
        value_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        omega: Union[float, np.ndarray, ArrayLike] = -4.0,
        rho: Union[float, np.ndarray, ArrayLike] = 0.0,
        additional_parameters: Optional[Dict] = None,
    ):
        """Add a value parent to a given set of nodes.

        Parameters
        ----------
        children_idxs :
            The child(s) node index(es).
        value_coupling :
            The value_coupling between the child and parent node. This is will be
            appended to the `psis` parameters in the parent and child node(s).
        mu :
            The mean of the Gaussian distribution. This value is passed both to the
            current and expected states.
        pi :
            The precision of the Gaussian distribution (inverse variance). This
            value is passed both to the current and expected states.
        omega :
            The tonic part of the variance (the variance that is not inherited from
            volatility parent(s)).
        rho :
            The drift of the random walk. Defaults to `0.0` (no drift).
        additional_parameters :
            Add more custom parameters to the node.

        """
        if not isinstance(children_idxs, list):
            children_idxs = [children_idxs]

        # how many nodes in structure
        n_nodes = len(self.edges)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu,
            "pi": pi,
            "pihat": pi,
            "kappas_children": None,
            "kappas_parents": None,
            "psis_children": tuple(value_coupling for _ in range(len(children_idxs))),
            "psis_parents": None,
            "omega": omega,
            "rho": rho,
        }

        # add more parameters (optional)
        if additional_parameters is not None:
            node_parameters = {**node_parameters, **additional_parameters}

        # add a new node to connection structure with no parents
        self.edges += (Indexes(None, None, tuple(children_idxs), None),)

        # add a new parameters to parameters structure
        self.attributes[parent_idx] = node_parameters

        # convert the structure to a list to modify it
        edges_as_list: List[Indexes] = list(self.edges)

        for idx in children_idxs:
            # add this node as parent and set value coupling
            if edges_as_list[idx].value_parents is not None:
                # append new index
                new_value_parents_idx = edges_as_list[idx].value_parents + (parent_idx,)
                edges_as_list[idx] = Indexes(
                    new_value_parents_idx,
                    edges_as_list[idx].volatility_parents,
                    edges_as_list[idx].value_children,
                    edges_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.attributes[idx]["psis_parents"] += (value_coupling,)
            else:
                # append new index
                new_value_parents_idx = (parent_idx,)
                edges_as_list[idx] = Indexes(
                    new_value_parents_idx,
                    edges_as_list[idx].volatility_parents,
                    edges_as_list[idx].value_children,
                    edges_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.attributes[idx]["psis_parents"] = (value_coupling,)

        # convert the list back to a tuple
        self.edges = tuple(edges_as_list)

        return self

    def add_volatility_parent(
        self,
        children_idxs: Union[List, int],
        volatility_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        omega: Union[float, np.ndarray, ArrayLike] = -4.0,
        rho: Union[float, np.ndarray, ArrayLike] = 0.0,
        additional_parameters: Optional[Dict] = None,
    ):
        """Add a volatility parent to a given set of nodes.

        Parameters
        ----------
        children_idxs :
            The child(s) node index(es).
        volatility_coupling :
            The volatility coupling between the child and parent node. This is will be
            appended to the `kappas` parameters in the parent and child node(s).
        mu :
            The mean of the Gaussian distribution. This value is passed both to the
            current and expected states.
        pi :
            The precision of the Gaussian distribution (inverse variance). This
            value is passed both to the current and expected states.
        omega :
            The tonic part of the variance (the variance that is not inherited from
            volatility parent(s)).
        rho :
            The drift of the random walk. Defaults to `0.0` (no drift).
        additional_parameters :
            Add more custom parameters to the node.

        """
        if not isinstance(children_idxs, list):
            children_idxs = [children_idxs]

        # how many nodes in structure
        n_nodes = len(self.edges)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu,
            "pi": pi,
            "pihat": pi,
            "kappas_children": tuple(
                volatility_coupling for _ in range(len(children_idxs))
            ),
            "kappas_parents": None,
            "psis_children": None,
            "psis_parents": None,
            "omega": omega,
            "rho": rho,
        }

        # add more parameters (optional)
        if additional_parameters is not None:
            node_parameters = {**node_parameters, **additional_parameters}

        # add a new node to the connection structure with no parents
        self.edges += (Indexes(None, None, None, tuple(children_idxs)),)

        # add a new parameter to the parameters structure
        self.attributes[parent_idx] = node_parameters

        # convert the structure to a list to modify it
        edges_as_list: List[Indexes] = list(self.edges)

        for idx in children_idxs:
            # add this node as a parent and set value coupling
            if edges_as_list[idx].volatility_parents is not None:
                # append new index
                new_volatility_parents_idx = edges_as_list[idx].volatility_parents + (
                    parent_idx,
                )
                edges_as_list[idx] = Indexes(
                    edges_as_list[idx].value_parents,
                    new_volatility_parents_idx,
                    edges_as_list[idx].value_children,
                    edges_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.attributes[idx]["kappas_parents"] += (volatility_coupling,)
            else:
                # append new index
                new_volatility_parents_idx = (parent_idx,)
                edges_as_list[idx] = Indexes(
                    edges_as_list[idx].value_parents,
                    new_volatility_parents_idx,
                    edges_as_list[idx].value_children,
                    edges_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.attributes[idx]["kappas_parents"] = (volatility_coupling,)

        # convert the list back to a tuple
        self.edges = tuple(edges_as_list)

        return self

    def set_update_sequence(self) -> "HGF":
        """Generate an update sequence from the network's structure.

        See :py:func:`pyhgf.networks.get_update_sequence` for more details.

        """
        update_sequence, prediction_sequence = get_update_sequence(
            hgf=self, update_sequence=[], prediction_sequence=[]
        )
        # generate the prediction sequence here, in reverse order
        prediction_sequence.reverse()
        prediction_sequence.extend(update_sequence)
        self.update_sequence = tuple(prediction_sequence)

        return self
