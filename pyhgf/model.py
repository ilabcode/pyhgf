# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import scan, switch
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.plots import plot_correlations, plot_network, plot_nodes, plot_trajectories
from pyhgf.response import first_level_binary_surprise, first_level_gaussian_surprise
from pyhgf.structure import beliefs_propagation
from pyhgf.typing import Indexes, InputIndexes, NodeStructure, UpdateSequence
from pyhgf.updates.binary import binary_input_update, binary_node_update
from pyhgf.updates.continuous import (
    continuous_input_update,
    continuous_node_update,
    gaussian_surprise,
)


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
    node_structure :
        A tuple of :py:class:`pyhgf.typing.Indexes` representing the nodes hierarchy.
    node_trajectories :
        The dynamic of the node's beliefs after updating.
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
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
        kappas: Dict = {"1": 1.0, "2": 0.0},
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
        self.node_structure: NodeStructure
        self.node_trajectories: Dict
        self.parameters_structure: Dict
        self.update_sequence: Optional[UpdateSequence] = None

        if model_type in ["continuous", "binary"]:
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
                    kind="continuous", continuous_precision=continuous_precision
                )
            elif model_type == "binary":
                self.add_input_node(
                    kind="binary",
                    eta0=eta0,
                    eta1=eta1,
                    binary_precision=binary_precision,
                )

            #########
            # x - 1 #
            #########
            self.add_value_parent(
                children_idxs=[0],
                value_coupling=1.0,
                mu=initial_mu["1"],
                pi=initial_pi["1"],
                omega=omega["1"] if self.model_type != "binary" else np.nan,
                rho=rho["1"] if self.model_type != "binary" else np.nan,
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

            # get the update sequence from structure
            self.update_sequence = self.get_update_sequence()

    def input_data(
        self,
        input_data: np.ndarray,
        time_steps: Optional[np.ndarray] = None,
    ):
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
        if self.update_sequence is None:
            self.update_sequence = self.get_update_sequence()

        # create the function that will be scaned
        scan_fn = Partial(
            beliefs_propagation,
            update_sequence=self.update_sequence,
            node_structure=self.node_structure,
            input_nodes_idx=self.input_nodes_idx.idx,
        )

        # concatenate data and time
        time_steps = time_steps[..., jnp.newaxis]
        if input_data.ndim == 1:
            input_data = input_data[..., jnp.newaxis]

        data = jnp.concatenate((input_data, time_steps), dtype=float, axis=1)

        # Run the entire for loop
        # this is where the model loop over the input time series
        # at each input the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        _, node_trajectories = scan(scan_fn, self.parameters_structure, data)

        # the node structure at each value update
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
                node_structure=self.node_structure,
                input_nodes_idx=self.input_nodes_idx.idx,
            )
            for seq in update_branches
        ]

        # create the function that will be scanned
        def switching_propagation(parameters_structure, scan_input):
            data, idx = scan_input
            return switch(idx, branches_fn, parameters_structure, data)

        # wrap the inputs
        scan_input = data, branches_idx

        # scan over the input data and apply the switching belief propagation functions
        _, node_trajectories = scan(
            switching_propagation, self.parameters_structure, scan_input
        )

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
        # get time and time steps from the first input node
        structure_df = pd.DataFrame(
            {
                "time_steps": self.node_trajectories[self.input_nodes_idx.idx[0]][
                    "time_step"
                ],
                "time": jnp.cumsum(
                    self.node_trajectories[self.input_nodes_idx.idx[0]]["time_step"]
                ),
            }
        )

        # add the observations from input nodes
        for inpt in self.input_nodes_idx.idx:
            structure_df[f"observation_input_{inpt}"] = self.node_trajectories[inpt][
                "value"
            ]

        # loop over nodes and store sufficient statistics with surprise
        for i in range(1, len(self.node_trajectories)):
            if i not in self.input_nodes_idx.idx:
                structure_df[f"x_{i}_mu"] = self.node_trajectories[i]["mu"]
                structure_df[f"x_{i}_pi"] = self.node_trajectories[i]["pi"]
                structure_df[f"x_{i}_muhat"] = self.node_trajectories[i]["muhat"]
                structure_df[f"x_{i}_pihat"] = self.node_trajectories[i]["pihat"]
                if (i == 1) & (self.model_type == "continuous"):
                    surprise = gaussian_surprise(
                        x=self.node_trajectories[0]["value"][1:],
                        muhat=self.node_trajectories[i]["muhat"][:-1],
                        pihat=self.node_trajectories[i]["pihat"][:-1],
                    )
                else:
                    surprise = gaussian_surprise(
                        x=self.node_trajectories[i]["mu"][1:],
                        muhat=self.node_trajectories[i]["muhat"][:-1],
                        pihat=self.node_trajectories[i]["pihat"][:-1],
                    )
                # fill with nans when the model cannot fit
                surprise = jnp.where(
                    jnp.isnan(self.node_trajectories[i]["mu"][1:]), jnp.nan, surprise
                )
                structure_df[f"x_{i}_surprise"] = np.insert(surprise, 0, np.nan)

        # compute the global surprise over all node
        structure_df["surprise"] = structure_df.iloc[
            :, structure_df.columns.str.contains("_surprise")
        ].sum(axis=1, min_count=1)

        return structure_df

    def add_input_node(
        self,
        kind: str,
        input_idxs: Union[List, int] = 0,
        continuous_precision: Union[float, np.ndarray, ArrayLike] = 1e4,
        binary_precision: Union[float, np.ndarray, ArrayLike] = jnp.inf,
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
        additional_parameters: Optional[Dict] = None,
    ):
        """Create an input node.

        Parameters
        ----------
        kind :
            The kind of input that should be created (can be `"continuous"` or
            `"binary"`).
        input_idxs :
            The index of the new input (defaults to `0`).
        continuous_precision :
            The continuous input precision (only relevant if `kind="continuous"`).
            Defaults to `1e4`.
        binary_precision :
            The binary input precision (only relevant if `kind="binary"`). Default to
            `jnp.inf`.
        eta0 :
            The lower bound of the binary process (only relevant if `kind="binary"`).
        eta1 :
            The lower bound of the binary process (only relevant if `kind="binary"`).
        additional_parameters :
            Add more custom parameters to the input node.

        """
        if isinstance(input_idxs, int):
            input_idxs = [input_idxs]

        if kind == "continuous":
            input_node_parameters = {
                "kappas_parents": None,
                "pihat": continuous_precision,
                "time_step": jnp.nan,
                "value": jnp.nan,
            }
        elif kind == "binary":
            input_node_parameters = {
                "pihat": binary_precision,
                "eta0": eta0,
                "eta1": eta1,
                "surprise": jnp.nan,
                "time_step": jnp.nan,
                "value": jnp.nan,
            }

        # add more parameters (optional)
        if additional_parameters is not None:
            input_node_parameters = {**input_node_parameters, **additional_parameters}

        for input_idx in input_idxs:
            if input_idx == 0:
                # this is the first node, create the node structure
                self.parameters_structure = {input_idx: input_node_parameters}
                self.node_structure = (Indexes(None, None, None, None),)
                self.input_nodes_idx = InputIndexes((input_idx,), (kind,))
            else:
                # update the node structure
                self.parameters_structure[input_idx] = input_node_parameters
                self.node_structure += (Indexes(None, None, None, None),)

                # add information about the new input node in the indexes
                new_idx = self.input_nodes_idx.idx
                new_idx += (input_idx,)
                new_kind = self.input_nodes_idx.kind
                new_kind += (kind,)
                self.input_nodes_idx = InputIndexes(new_idx, new_kind)

        return self

    def add_value_parent(
        self,
        children_idxs: Union[List, int],
        value_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        mu_hat: Union[float, np.ndarray, ArrayLike] = 0.0,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        pi_hat: Union[float, np.ndarray, ArrayLike] = 1.0,
        nu: Union[float, np.ndarray, ArrayLike] = jnp.nan,
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
            The mean of the Gaussian distribution.
        mu_hat :
            The expected mean of the Gaussian distribution for the next observation.
        pi :
            The precision of the Gaussian distribution (inverse variance).
        pi_hat :
            The expected precision of the Gaussian distribution (inverse variance) for
            the next observation.
        nu :
            Stochasticity coupling.
        omega :
            The tonic part of the variance (the variance that is not inherited from
            volatility parent(s)).
        rho :
            The drift of the random walk. Defaults to `0.0` (no drift).
        additional_parameters :
            Add more custom parameters to the node.

        """
        if isinstance(children_idxs, int):
            children_idxs = [children_idxs]

        # how many nodes in structure
        n_nodes = len(self.node_structure)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu_hat,
            "pi": pi,
            "pihat": pi_hat,
            "kappas_children": None,
            "kappas_parents": None,
            "psis_children": tuple(value_coupling for _ in range(len(children_idxs))),
            "psis_parents": None,
            "nu": nu,
            "omega": omega,
            "rho": rho,
        }

        # add more parameters (optional)
        if additional_parameters is not None:
            node_parameters = {**node_parameters, **additional_parameters}

        # add a new node to connection structure with no parents
        self.node_structure += (Indexes(None, None, tuple(children_idxs), None),)

        # add a new parameters to parameters structure
        self.parameters_structure[parent_idx] = node_parameters

        # convert the structure to a list to modify it
        structure_as_list: List[Indexes] = list(self.node_structure)

        for idx in children_idxs:
            # add this node as parent and set value coupling
            if structure_as_list[idx].value_parents is not None:
                # append new index
                new_value_parents_idx = structure_as_list[idx].value_parents + (
                    parent_idx,
                )
                structure_as_list[idx] = Indexes(
                    new_value_parents_idx,
                    structure_as_list[idx].volatility_parents,
                    structure_as_list[idx].value_children,
                    structure_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.parameters_structure[idx]["psis_parents"] += (value_coupling,)
            else:
                # append new index
                new_value_parents_idx = (parent_idx,)
                structure_as_list[idx] = Indexes(
                    new_value_parents_idx,
                    structure_as_list[idx].volatility_parents,
                    structure_as_list[idx].value_children,
                    structure_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.parameters_structure[idx]["psis_parents"] = (value_coupling,)

        # convert the list back to a tuple
        self.node_structure = tuple(structure_as_list)

        return self

    def add_volatility_parent(
        self,
        children_idxs: Union[List, int],
        volatility_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        mu_hat: Union[float, np.ndarray, ArrayLike] = 0.0,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        pi_hat: Union[float, np.ndarray, ArrayLike] = 1.0,
        nu: Union[float, np.ndarray, ArrayLike] = jnp.nan,
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
            The mean of the Gaussian distribution.
        mu_hat :
            The expected mean of the Gaussian distribution for the next observation.
        pi :
            The precision of the Gaussian distribution (inverse variance).
        pi_hat :
            The expected precision of the Gaussian distribution (inverse variance) for
            the next observation.
        nu :
            Stochasticity coupling.
        omega :
            The tonic part of the variance (the variance that is not inherited from
            volatility parent(s)).
        rho :
            The drift of the random walk. Defaults to `0.0` (no drift).
        additional_parameters :
            Add more custom parameters to the node.

        """
        if isinstance(children_idxs, int):
            children_idxs = [children_idxs]

        # how many nodes in structure
        n_nodes = len(self.node_structure)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu_hat,
            "pi": pi,
            "pihat": pi_hat,
            "kappas_children": tuple(
                volatility_coupling for _ in range(len(children_idxs))
            ),
            "kappas_parents": None,
            "psis_children": None,
            "psis_parents": None,
            "nu": nu,
            "omega": omega,
            "rho": rho,
        }

        # add more parameters (optional)
        if additional_parameters is not None:
            node_parameters = {**node_parameters, **additional_parameters}

        # add a new node to the connection structure with no parents
        self.node_structure += (Indexes(None, None, None, tuple(children_idxs)),)

        # add a new parameter to the parameters structure
        self.parameters_structure[parent_idx] = node_parameters

        # convert the structure to a list to modify it
        structure_as_list: List[Indexes] = list(self.node_structure)

        for idx in children_idxs:
            # add this node as a parent and set value coupling
            if structure_as_list[idx].volatility_parents is not None:
                # append new index
                new_volatility_parents_idx = structure_as_list[
                    idx
                ].volatility_parents + (parent_idx,)
                structure_as_list[idx] = Indexes(
                    structure_as_list[idx].value_parents,
                    new_volatility_parents_idx,
                    structure_as_list[idx].value_children,
                    structure_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.parameters_structure[idx]["kappas_parents"] += (
                    volatility_coupling,
                )
            else:
                # append new index
                new_volatility_parents_idx = (parent_idx,)
                structure_as_list[idx] = Indexes(
                    structure_as_list[idx].value_parents,
                    new_volatility_parents_idx,
                    structure_as_list[idx].value_children,
                    structure_as_list[idx].volatility_children,
                )
                # set the value coupling strength
                self.parameters_structure[idx]["kappas_parents"] = (
                    volatility_coupling,
                )

        # convert the list back to a tuple
        self.node_structure = tuple(structure_as_list)

        return self

    def get_update_sequence(
        self, node_idxs: Tuple[int, ...] = (0,), update_sequence: Tuple = ()
    ) -> Tuple:
        """Automated creation of the update sequence from the network's structure.

        The function ensures that the following rules apply:
        1. all children have been updated before propagating to the parent(s) node.
        2. the update function of an input node is chosen based on the node's type
        (`"continuous"` or `"binary"`).
        3. the update function of the parent of an input node is chosen based on the
        node's type (`"continuous"` or `"binary"`).

        Parameters
        ----------
        node_idxs :
            The indexes of the current node(s) that should be evaluated. By default
            (`None`), the function will start with the list of input nodes.
        update_sequence :
            The current state of the update sequence (for recursive evaluation).

        Returns
        -------
        update_sequence :
            The update sequence.

        """
        if len(update_sequence) == 0:
            node_idxs = self.input_nodes_idx.idx

        for node_idx in node_idxs:
            # if the node has no parent, exit here
            if (self.node_structure[node_idx].value_parents is None) & (
                self.node_structure[node_idx].volatility_parents is None
            ):
                continue

            # select the update function
            # --------------------------

            # case 1 - default to a continuous node
            update_fn = continuous_node_update

            # case 2 - this is an input node
            if node_idx in self.input_nodes_idx.idx:
                model_kind = [
                    kind_
                    for kind_, idx_ in zip(
                        self.input_nodes_idx.kind, self.input_nodes_idx.idx
                    )
                    if idx_ == node_idx
                ][0]
                if model_kind == "binary":
                    update_fn = binary_input_update
                elif model_kind == "continuous":
                    update_fn = continuous_input_update

            # case 3 - this is the value parent of at least one binary input node
            if self.node_structure[node_idx].value_children is not None:
                value_children_idx = self.node_structure[node_idx].value_children
                for child_idx in value_children_idx:  # type: ignore
                    if child_idx in self.input_nodes_idx.idx:
                        model_kind = [
                            kind_
                            for kind_, idx_ in zip(
                                self.input_nodes_idx.kind, self.input_nodes_idx.idx
                            )
                            if idx_ == child_idx
                        ][0]
                        if model_kind == "binary":
                            update_fn = binary_node_update

            # create a new sequence step and add it to the list
            new_sequence = node_idx, update_fn
            update_sequence += (new_sequence,)

            # search recursively for the next update steps - make sure that all the
            # children have been updated before updating the parent(s)
            # ---------------------------------------------------------------------

            # loop over all possible dependencies (e.g. value, volatility coupling)
            parents = self.node_structure[node_idx][:2]
            for parent_types in parents:
                if parent_types is not None:
                    for idx in parent_types:
                        # get the list of all the children in the parent's family
                        children_idxs = []
                        if self.node_structure[idx].value_children is not None:
                            children_idxs.extend(
                                [*self.node_structure[idx].value_children]
                            )
                        if self.node_structure[idx].volatility_children is not None:
                            children_idxs.extend(
                                [*self.node_structure[idx].volatility_children]
                            )

                        if len(children_idxs) > 0:
                            # only call the update on the parent(s) if the current node
                            # is the last on the children's list (meaning that all the
                            # other nodes have been updated)
                            if children_idxs[-1] == node_idx:
                                update_sequence = self.get_update_sequence(
                                    node_idxs=(idx,), update_sequence=update_sequence
                                )

        return update_sequence
