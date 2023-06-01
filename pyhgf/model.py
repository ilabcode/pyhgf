# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from math import log
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.lax import scan
from jax.tree_util import Partial
from jax.typing import ArrayLike

from pyhgf.binary import binary_input_update, binary_node_update
from pyhgf.continuous import (
    continuous_input_update,
    continuous_node_update,
    gaussian_surprise,
)
from pyhgf.plots import plot_correlations, plot_network, plot_trajectories
from pyhgf.response import first_level_binary_surprise, first_level_gaussian_surprise
from pyhgf.structure import loop_inputs
from pyhgf.typing import Indexes, InputIndexes, NodeStructure, UpdateSequence


class HGF(object):
    r"""The two-level and three-level Hierarchical Gaussian Filters (HGF).

    This class use pre-made nodes structures that corresponds to the most widely used
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
        The parameter structure that incluse the consecutive updates at each new
        observation.
    parameters_structure :
        The structure of nodes' parameters. Each parameter is a dictionary with the
        following parameters: `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"` for
        continuous nodes.
    .. note::
        `"psis"` is the value coupling strength. It should have same length than the
        volatility parents' indexes.
        `"kappas"` is the volatility coupling strength. It should have same length
        than the volatility parents' indexes.
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
        omega_input: Union[float, np.ndarray, ArrayLike] = log(1e-4),
        omega: Dict = {
            "1": -3.0,
            "2": -3.0,
            "3": -3.0,
        },
        kappas: Dict = {"1": 1.0, "2": 0.0},
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
        pihat: Union[float, np.ndarray, ArrayLike] = jnp.inf,
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
            Dictionary containing the initial values for the :math:`\\mu` parameter at
            different levels of the hierarchy. Defaults set to `0.0` at all levels.
        initial_pi :
            Dictionary containing the initial values for the :math:`\\pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults set to `1.0` at all levels.
        kappas :
            Dictionary containing the initial values for the :math:`\\kappa` parameter
            at different levels of the hierarchy. Kappa represents the phasic part of
            the variance (the part that is affected by the parents nodes) and will
            defines the strenght of the connection between the node and the parent
            node. Often fixed to `1`. Defaults set to `1` at all levels.
        model_type : str
            The model type to use (can be `"continuous"` or `"binary"`).
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Defaults to `2` for a 2-level HGF.
        omega :
            Dictionary containing the initial values for the :math:`\\omega` parameter
            at different levels of the hierarchy. :math:`\\omega` represents the tonic
            part of the variance (the part that is not affected by the parent node).
            Defaults set to `-3.0` at all levels. This parameter is only required for
            Gaussian Random Walk nodes.
        omega_input :
            Default value sets to `log(1e-4)`. Represents the noise associated with
            the input.
        pihat :
            The precision of the binary input node. Default to `jnp.inf`. Only relevant
            if `model_type="binary"`.
        rho :
            Dictionary containing the initial values for the :math:`\\rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random
            walk. This parameter is only required for Gaussian Random Walk nodes.
            Defaults set all entries to `0` according to the number of required levels.
        verbose :
            Verbosity of the methods for model creation and fitting. Defaults to `True`.

        """
        self.model_type = model_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.node_structure: NodeStructure
        self.node_trajectories: Dict
        self.parameters_structure: Dict
        self.input_nodes_idx: InputIndexes = InputIndexes([], [])
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
                self.add_input_node(kind="continuous", omega_input=omega_input)
            elif model_type == "binary":
                self.add_input_node(kind="binary", eta0=eta0, eta1=eta1, pihat=pihat)

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
        time: Optional[np.ndarray] = None,
    ):
        """Add new observations.

        Parameters
        ----------
        input_data :
            The new observations.
        time :
            Time vector (optional). If `None`, the time vector will defaults to
            `np.arange(0, len(input_data))`.

        """
        if self.verbose:
            print((f"Adding {len(input_data)} new observations."))
        if time is None:
            time = np.ones(len(input_data), dtype=int)  # time step vector
        if self.update_sequence is None:
            self.update_sequence = self.get_update_sequence()

        # create the function that will be scaned
        scan_fn = Partial(
            loop_inputs,
            update_sequence=self.update_sequence,
            node_structure=self.node_structure,
            input_nodes_idx=jnp.array(self.input_nodes_idx.idx),
        )

        # create the input array (data, time)
        data = jnp.array([input_data, time], dtype=float).T

        # Run the entire for loop
        # this is where the model loop over the input time series
        # at each input the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        _, node_trajectories = scan(scan_fn, self.parameters_structure, data)

        # the node structure at each value update
        self.node_trajectories = node_trajectories

        return self

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
            (default), return the sum of gaussian surprise if `model_type=="continuous"`
            or the sum of the binary surprise if `model_type=="binary"`.
        response_function_parameters :
            Additional parameters to the response function (optional) .

        Returns
        -------
        surprise :
            The model surprise given the input data and the response function.

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
            Pandas data frame with the time series of the sufficient statistics and
            surprise of each node in the structure.

        """
        # input parameters
        structure_df = pd.DataFrame(
            {
                "time_steps": self.node_trajectories[0]["time_step"],
                "time": jnp.cumsum(self.node_trajectories[0]["time_step"]),
                "observation": self.node_trajectories[0]["value"],
            }
        )

        # loop over nodes and store sufficient statistics with surprise
        for i in range(1, len(self.node_trajectories)):
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
        input_idx: int = 0,
        omega_input: Union[float, np.ndarray, ArrayLike] = log(1e-4),
        pihat: Union[float, np.ndarray, ArrayLike] = jnp.inf,
        eta0: Union[float, np.ndarray, ArrayLike] = 0.0,
        eta1: Union[float, np.ndarray, ArrayLike] = 1.0,
    ):
        """Create an input node.

        Parameters
        ----------
        kind :
            The kind of input that should be created (can be `"continuous"` or
            `"binary"`).
        input_idx :
            The index of the new input (defaults to `0`).
        omega_input :
            The input precision (only relevant if `kind="continuous"`).
        pihat :
            The input precision (only relevant if `kind="binary"`).
        eta0 :
            The lower bound of the binary process (only relevant if `kind="binary"`).
        eta1 :
            The lower bound of the binary process (only relevant if `kind="binary"`).

        """
        if kind == "continuous":
            input_node_parameters = {
                "kappas": None,
                "omega": omega_input,
                "time_step": jnp.nan,
                "value": jnp.nan,
            }
        elif kind == "binary":
            input_node_parameters = {
                "pihat": pihat,
                "eta0": eta0,
                "eta1": eta1,
                "surprise": jnp.nan,
                "time_step": jnp.nan,
                "value": jnp.nan,
            }
        if input_idx == 0:
            # this is the first node, create the node structure
            self.parameters_structure = {input_idx: input_node_parameters}
            self.node_structure = (Indexes(None, None),)
        else:
            # update the node structure
            self.parameters_structure[input_idx] = input_node_parameters
            self.node_structure += (Indexes(None, None),)

        # add information about the new input node in the indexes
        self.input_nodes_idx.idx.append(input_idx)
        self.input_nodes_idx.kind.append(kind)

        return self

    def add_value_parent(
        self,
        children_idxs: List,
        value_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        mu_hat: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        pi_hat: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        kappas: Optional[Tuple] = None,
        nu: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        psis: Optional[Tuple] = None,
        omega: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        rho: Union[float, np.ndarray, ArrayLike] = 0.0,
    ):
        """Add a value parent to node.

        Parameters
        ----------
        children_idxs :
            The child(s) node index(es).
        value_coupling :
            The value_coupling between the child and parent node. This is will be
            appended to the `psis` parameters in the child node.
        mu :
            Mean.
        mu_hat :
            Expected mean.
        pi :
            Precision.
        pi_hat :
            Expected precision.
        kappas :
            Phasic part of the variance.
        nu :
            Stochasticity coupling.
        psis :
            Value coupling.
        omega :
            Tonic part of the variance.
        rho :
            Drift of the random walk.

        """
        # how many nodes in structure
        n_nodes = len(self.node_structure)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu_hat,
            "pi": pi,
            "pihat": pi_hat,
            "kappas": kappas,
            "nu": nu,
            "psis": psis,
            "omega": omega,
            "rho": rho,
        }

        # add a new node to connection structure with no parents
        self.node_structure += (Indexes(None, None),)

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
                    new_value_parents_idx, structure_as_list[idx].volatility_parents
                )
                # set the value coupling strength
                self.parameters_structure[idx]["psis"] += (value_coupling,)
            else:
                # append new index
                new_value_parents_idx = (parent_idx,)
                structure_as_list[idx] = Indexes(
                    new_value_parents_idx, structure_as_list[idx].volatility_parents
                )
                # set the value coupling strength
                self.parameters_structure[idx]["psis"] = (value_coupling,)

        # convert the list back to a tuple
        self.node_structure = tuple(structure_as_list)

        return self

    def add_volatility_parent(
        self,
        children_idxs: List,
        volatility_coupling: Union[float, np.ndarray, ArrayLike] = 1.0,
        mu: Union[float, np.ndarray, ArrayLike] = 0.0,
        mu_hat: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        pi: Union[float, np.ndarray, ArrayLike] = 1.0,
        pi_hat: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        kappas: Optional[Tuple] = None,
        nu: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        psis: Optional[Tuple] = None,
        omega: Union[float, np.ndarray, ArrayLike] = jnp.nan,
        rho: Union[float, np.ndarray, ArrayLike] = 0.0,
    ):
        """Add a volatility parent to node.

        Parameters
        ----------
        children_idxs :
            The child(s) node index(es).
        volatility_coupling :
            The volatility coupling between the child and parent node. This is will be
            appended to the `kappas` parameters in the child node.
        mu :
            Mean.
        mu_hat :
            Expected mean.
        pi :
            Precision.
        pi_hat :
            Expected precision.
        kappas :
            Phasic part of the variance.
        nu :
            Stochasticity coupling.
        psis :
            Value coupling.
        omega :
            Tonic part of the variance.
        rho :
            Drift of the random walk.

        """
        # how many nodes in structure
        n_nodes = len(self.node_structure)
        parent_idx = n_nodes  # append a new parent in the structure

        # parent's parameter
        node_parameters = {
            "mu": mu,
            "muhat": mu_hat,
            "pi": pi,
            "pihat": pi_hat,
            "kappas": kappas,
            "nu": nu,
            "psis": psis,
            "omega": omega,
            "rho": rho,
        }

        # add a new node to connection structure with no parents
        self.node_structure += (Indexes(None, None),)

        # add a new parameters to parameters structure
        self.parameters_structure[parent_idx] = node_parameters

        # convert the structure to a list to modify it
        structure_as_list: List[Indexes] = list(self.node_structure)

        for idx in children_idxs:
            # add this node as parent and set value coupling
            if structure_as_list[idx].volatility_parents is not None:
                # append new index
                new_volatility_parents_idx = structure_as_list[
                    idx
                ].volatility_parents + (parent_idx,)
                structure_as_list[idx] = Indexes(
                    structure_as_list[idx].value_parents, new_volatility_parents_idx
                )
                # set the value coupling strength
                self.parameters_structure[idx]["kappas"] += (volatility_coupling,)
            else:
                # append new index
                new_volatility_parents_idx = (parent_idx,)
                structure_as_list[idx] = Indexes(
                    structure_as_list[idx].value_parents, new_volatility_parents_idx
                )
                # set the value coupling strength
                self.parameters_structure[idx]["kappas"] = (volatility_coupling,)

        # convert the list back to a tuple
        self.node_structure = tuple(structure_as_list)

        return self

    def get_update_sequence(
        self, node_idx: int = 0, update_sequence: Tuple = ()
    ) -> Tuple:
        """Get update sequence from the nodes' connections structure.

        Parameters
        ----------
        node_idx :
            The index of the current node.
        update_sequence :
            The current update sequence. This is used for recursive evaluation.

        Returns
        -------
        update_sequence :
            The update sequence.

        """
        # get the parents indexes
        value_parent_idx = self.node_structure[node_idx].value_parents
        volatility_parent_idx = self.node_structure[node_idx].volatility_parents

        # select the update function
        if node_idx in self.input_nodes_idx.idx:
            if self.model_type == "binary":
                update_fn = binary_input_update
            elif self.model_type == "continuous":
                update_fn = continuous_input_update
        elif (node_idx == 1) & (self.model_type == "binary"):
            update_fn = binary_node_update
        else:
            update_fn = continuous_node_update

        # create a new sequence step and add it to the list
        new_sequence = node_idx, update_fn
        update_sequence += (new_sequence,)

        # search recursively for the nex update steps
        if value_parent_idx is not None:
            for idx in value_parent_idx:
                update_sequence = self.get_update_sequence(
                    node_idx=idx, update_sequence=update_sequence
                )
        if volatility_parent_idx is not None:
            for idx in volatility_parent_idx:
                update_sequence = self.get_update_sequence(
                    node_idx=idx, update_sequence=update_sequence
                )

        return update_sequence
