# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from math import log
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.interpreters.xla import DeviceArray
from jax.lax import scan

from pyhgf.binary import loop_binary_inputs
from pyhgf.continuous import gaussian_surprise, loop_continuous_inputs
from pyhgf.plots import plot_correlations, plot_trajectories
from pyhgf.response import total_binary_surprise, total_gaussian_surprise
from pyhgf.structure import structure_as_dict


class HGF(object):
    r"""The two-level and three-level Hierarchical Gaussian Filters (HGF).

    This class use pre-made nodes structures that corresponds to the most widely used
    HGF. The inputs can be continuous or binary.

    Attributes
    ----------
    hgf_results : dict
        After oberving the data using the `input_data` method, the output of the model
        are stored in the `hgf_results` dictionary.
    model_type : str
        The model implemented (can be `"continuous"`, `"binary"` or `"custom"`).
    n_levels : int
        The number of hierarchies in the model, including the input vector. Cannot be
        less than 2.
    node_structure : tuple
        A tuple of tuples representing the nodes hierarchy.
    node_trajectories : tuples
        The node structure updated at each new observation.
    results :
        Time, values inputs and overall surprise of the model.
    structure_dict : dict
        The node structure as a dictionary,
        retrieved using :py:func:`pyhgf.structure.structure_as_dict`
    verbose : bool
        Verbosity level.

    """

    def __init__(
        self,
        n_levels: Optional[int] = 2,
        model_type: str = "continuous",
        initial_mu: Dict[str, Optional[float]] = {"1": 0.0, "2": 0.0},
        initial_pi: Dict[str, Optional[float]] = {"1": 1.0, "2": 1.0},
        omega_input: float = log(1e-4),
        omega: Dict[str, Optional[float]] = {"1": -3.0, "2": -3.0},
        kappas: Dict[str, Optional[float]] = {"1": 1.0},
        eta0: float = 0.0,
        eta1: float = 1.0,
        pihat: float = jnp.inf,
        rho: Dict[str, Optional[float]] = {"1": 0.0, "2": 0.0},
        verbose: bool = True,
    ):
        r"""Parameterization of the HGF model.

        Parameters
        ----------
        n_levels :
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Defaults to `2` for a 2-level HGF.
        model_type : str
            The model type to use (can be `"continuous"` or `"binary"`).
        initial_mu :
            Dictionary containing the initial values for the :math:`\\mu` parameter at
            different levels of the hierarchy. Defaults set to `{"1": 0.0, "2": 0.0}`
            for a 2-level model for continuous inputs.
        initial_pi :
            Dictionary containing the initial values for the :math:`\\pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults set to `{"1": 1.0, "2": 1.0}` for
            a 2-level model.
        omega :
            Dictionary containing the initial values for the :math:`\\omega` parameter
            at different levels of the hierarchy. :math:`\\omega` represents the tonic
            part of the variance (the part that is not affected by the parent node).
            Defaults set to `{"1": -10.0, "2": -10.0}` for a 2-level model for
            continuous inputs. This parameter is only required for Gaussian Random Walk
            nodes.
        omega_input :
            Default value sets to `log(1e-4)`. Represents the noise associated with
            the input.
        rho :
            Dictionary containing the initial values for the :math:`\\rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random
            walk. This parameter is only required for Gaussian Random Walk nodes.
            Defaults set all entries to `0` according to the number of required levels.
        kappas :
            Dictionary containing the initial values for the :math:`\\kappa` parameter
            at different levels of the hierarchy. Kappa represents the phasic part of
            the variance (the part that is affected by the parents nodes) and will
            defines the strenght of the connection between the node and the parent
            node. Often fixed to `1`. Defaults set to `{"1": 1.0}` for a 2-level model.
            This parameter is only required for Gaussian Random Walk nodes.
        eta0 :
            The first categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        eta1 :
            The second categorical value of the binary node. Defaults to `0.0`. Only
            relevant if `model_type="binary"`.
        pihat :
            The precision of the binary input node. Default to `jnp.inf`. Only relevant
            if `model_type="binary"`.
        verbose :
            Verbosity of the methods for model creation and fitting. Defaults to `True`.

        """
        self.model_type = model_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.structure_dict = None

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
                children_idx=[0],
                parent_idx=1,
                mu=initial_mu["1"],
                pi=initial_pi["1"],
                omega=omega["1"],
                rho=rho["1"],
            )

            #########
            # x - 2 #
            #########
            if model_type == "continuous":
                self.add_volatility_parent(
                    children_idx=[1],
                    parent_idx=2,
                    volatility_coupling=kappas["1"],
                    mu=initial_mu["2"],
                    pi=initial_pi["2"],
                    omega=omega["2"],
                    rho=rho["2"],
                )
            elif model_type == "binary":
                self.add_value_coupling(
                    children_idx=[1],
                    parent_idx=2,
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
                    children_idx=[2],
                    parent_idx=3,
                    volatility_coupling=kappas["2"],
                    mu=initial_mu["3"],
                    pi=initial_pi["3"],
                    omega=omega["3"],
                    rho=rho["3"],
                )

    def input_data(
        self,
        input_data: Union[List, np.ndarray, DeviceArray],
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
        input_data = jnp.asarray(input_data)

        if self.verbose:
            print((f"Add {input_data.shape[0]} new {self.model_type} observations."))
        if time is None:
            # create a time vector (starting at 1 as time 0 already exists by default)
            time = jnp.arange(0.0, len(input_data))
        else:
            assert len(input_data) == len(time)

        # Initialise the first values
        res_init = (
            self.node_structure,
            {
                "time": time[0],
                "value": input_data[0],
                "surprise": jnp.array(0.0),
            },
        )

        # create the input array (data, time)
        data = jnp.array([input_data, time], dtype=float).T

        # this is where the model loop over the input time series
        # at each input the node structure is traversed and beliefs are updated
        # using precision-weighted prediction errors
        if self.model_type == "continuous":
            _, scan_updates = scan(loop_continuous_inputs, res_init, data[1:, :])
        elif self.model_type == "binary":
            _, scan_updates = scan(loop_binary_inputs, res_init, data[1:, :])
        else:
            raise ValueError("This method only works with binary or continuous models")

        # the node structure at each value update
        self.node_trajectories = scan_updates[0]

        self.results = scan_updates[1]  # time, values and surprise

        return self

    def plot_trajectories(self, **kwargs):
        """Plot the parameters trajectories."""
        return plot_trajectories(hgf=self, **kwargs)

    def plot_correlations(self):
        """Plot the heatmap of cross-trajectories correlation."""
        return plot_correlations(hgf=self)

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
                total_gaussian_surprise
                if self.model_type == "continuous"
                else total_binary_surprise
            )

        return response_function(
            hgf=self,
            response_function_parameters=response_function_parameters,
        )

    def structure_as_dict(self) -> Dict:
        """Export the node structure to a dictionary of nodes.

        Returns
        -------
        structure_dict :
            The node structure as a dictionary, each node is a key in the dictionary,
            from 1 (lower node) to **n**.

        """
        if self.structure_dict is None:
            structure_dict = {}
            self.structure_dict = structure_as_dict(
                node_structure=self.node_trajectories,
                node_id=0,
                structure_dict=structure_dict,
            )
        return self.structure_dict

    def to_pandas(self) -> pd.DataFrame:
        """Export the nodes trajectories and surprise as a Pandas data frame.

        Returns
        -------
        structure_df :
            Pandas data frame with the time series of the sufficient statistics and
            surprise of each node in the structure.

        """
        node_dict = self.structure_as_dict()
        structure_df = pd.DataFrame(
            {
                "time": self.results["time"],
                "observation": self.results["value"],
                "surprise": self.results["surprise"],
            }
        )
        # loop over nodes and store sufficient statistics with surprise
        for key in list(node_dict.keys())[1:]:
            structure_df[f"{key}_mu"] = node_dict[key]["mu"]
            structure_df[f"{key}_pi"] = node_dict[key]["pi"]
            structure_df[f"{key}_muhat"] = node_dict[key]["muhat"]
            structure_df[f"{key}_pihat"] = node_dict[key]["pihat"]
            surprise = gaussian_surprise(
                x=node_dict[key]["mu"][1:],
                muhat=node_dict[key]["muhat"][:-1],
                pihat=node_dict[key]["pihat"][:-1],
            )
            structure_df[f"{key}_surprise"] = np.insert(surprise, 0, np.nan)

        return structure_df

    def add_input_node(
        self,
        kind: str,
        omega_input: float = log(1e-4),
        pihat: float = jnp.inf,
        eta0: float = 0.0,
        eta1: float = 1.0,
    ):
        """Create an input node."""
        if kind == "continuous":
            input_node_parameters = {"kind": kind, "kappas": None, "omega": omega_input}
        elif kind == "binary":
            input_node_parameters = {
                "kind": kind,
                "pihat": pihat,
                "eta0": eta0,
                "eta1": eta1,
            }
        self.node_structure = {
            0: {
                "parameters": input_node_parameters,
                "value_parents": None,
                "volatility_parents": None,
            }
        }
        return self

    def add_value_parent(
        self,
        children_idxs: List,
        value_coupling: float,
        parent_idx: Optional[int] = None,
        mu: float = 0.0,
        mu_hat: float = jnp.nan,
        pi: float = 1.0,
        pi_hat: float = jnp.nan,
        kappas: Optional[Tuple] = None,
        nu: float = jnp.nan,
        psis: Optional[Tuple] = None,
        omega: float = jnp.nan,
        rho: float = 0.0,
    ):
        """Add a value parent to node.

        children_idxs :
            The child(s) node index(s).
        parent idx :
            The new parent index.
        value_coupling :
            The value_coupling between the child and parent node. This is will be
            appended to the `psis` parameters in the child node.
        **kwargs :
            Node parameters.

        """
        # how many nodes in structure
        n_nodes = len(self.node_structure)

        if parent_idx is None:
            parent_idx = n_nodes + 1

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

        # add a new node with parameters
        self.node_structure[parent_idx] = {
            "parameters": node_parameters,
            "value_parents": None,
            "volatility_parents": None,
        }

        for idx in children_idxs:
            # add this node as parent and set value coupling
            if self.node_structure[idx]["value_parents"] is not None:
                self.node_structure[idx]["value_parents"] += (parent_idx,)
                self.node_structure[idx]["parameters"]["psis"] += (value_coupling,)
            else:
                self.node_structure[idx]["value_parents"] = (parent_idx,)
                self.node_structure[idx]["parameters"]["psis"] = (value_coupling,)

        return self

    def add_volatility_parent(
        self,
        children_idxs: List,
        volatility_coupling: float,
        parent_idx: Optional[int] = None,
        mu: float = 0.0,
        mu_hat: float = jnp.nan,
        pi: float = 1.0,
        pi_hat: float = jnp.nan,
        kappas: Optional[Tuple] = None,
        nu: float = jnp.nan,
        psis: Optional[Tuple] = None,
        omega: float = jnp.nan,
        rho: float = 0.0,
    ):
        """Add a volatility parent to node.

        node_idx :
            The child node(s) index.
        volatility_coupling :
            The volatility coupling between the child and parent node. This is will be
            appended to the `kappas` parameters in the child node.
        **kwargs :
            Node parameters.

        """
        # how many nodes in structure
        n_nodes = len(self.node_structure)

        if parent_idx is None:
            parent_idx = n_nodes + 1

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

        # add a new node with parameters
        self.node_structure[parent_idx] = {
            "parameters": node_parameters,
            "value_parents": None,
            "volatility_parents": None,
        }

        for idx in children_idxs:
            # add this node as parent and set value coupling
            if self.node_structure[idx]["value_parents"] is not None:
                self.node_structure[idx]["value_parents"] += (parent_idx,)
                self.node_structure[idx]["parameters"]["kappas"] += (
                    volatility_coupling,
                )
            else:
                self.node_structure[idx]["value_parents"] = (parent_idx,)
                self.node_structure[idx]["parameters"]["kappas"] = (
                    volatility_coupling,
                )

        return self
