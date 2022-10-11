# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from math import log
from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.lax import scan

from ghgf.jax import loop_inputs, node_validation
from ghgf.plots import plot_correlations, plot_trajectories
from ghgf.response import gaussian_surprise

NodeType = Tuple[
    Dict[str, Union[str, jnp.array, None]], Optional[Tuple], Optional[Tuple]
]


class HGF(object):
    """The standard 2 or 3 levels HGF for continuous inputs.

    Attributes
    ----------
    verbose : bool
        Verbosity level.
    n_levels : int
        The number of hierarchies in the model, including the input vector. Cannot be
        less than 2.
    model_type : str
        The model implemented (can be `"continous"` or `"binary"`).
    nodes : tuple
        A tuple of tuples representing the nodes hierarchy.
    hgf_results : dict
        After oberving the data using the `input_data` method, the output of the model
        are stored in the `hgf_results` dictionary.

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
        rho: Dict[str, Optional[float]] = {"1": 0.0, "2": 0.0},
        bias: float = 0.0,
        verbose: bool = True,
    ):

        """Parameterization of the HGF model.

        Parameters
        ----------
        n_levels : int | None
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Defaults to `2` for a 2-levels HGF.
        model_type : str
            The model type to use (can be "continuous" or "binary").
        initial_mu : dict
            Dictionary containing the initial values for the `initial_mu` parameter at
            different levels of the hierarchy. Defaults set to `{"1": 0.0, "2": 0.0}`
            for a 2-levels model.
        initial_pi : dict
            Dictionary containing the initial values for the `initial_pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults set to `{"1": 1.0, "2": 1.0}` for
            a 2-levels model.
        omega : dict
            Dictionary containing the initial values for the `omega` parameter at
            different levels of the hierarchy. Omegas represent the tonic part of the
            variance (the part that is not affected by the parent node). Defaults set to
            `{"1": -10.0, "2": -10.0}` for a 2-levels model. This parameters only when
            `model_type="GRW"`.
        omega_input : float
            Default value sets to `log(1e-4)`. Represents the noise associated with
            the input.
        rho : dict
            Dictionary containing the initial values for the `rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random
            walk. Only required when `model_type="GRW"`. Defaults set all entries to
            `0` according to the number of required levels.
        kappas : dict
            Dictionary containing the initial values for the `kappa` parameter at
            different levels of the hierarchy. Kappa represents the phasic part of the
            variance (the part that is affected by the parents nodes) and will defines
            the strenght of the connection between the node and the parent node. Often
            fixed to 1. Defaults set to `{"1": 1.0}` for a 2-levels model. Only
            required when `model_type="GRW"`.
        bias : DeviceArray
            The bias introduced in the perception of the input signal. This value is
            added to the input time serie before model fitting.
        verbose : bool
            Default is `True`.

        Raises
        ------
        ValueError
            If `model_type` is not `"continuous"` or `"binary"`.

        """

        self.model_type = model_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.bias = bias

        if model_type == "binary":

            if self.n_levels == 3:

                if self.verbose:
                    print(
                        (
                            "Fitting the binary Hierarchical Gaussian Filter (JAX) "
                            f"with {self.n_levels} levels."
                        )
                    )

                    # Third level
                    x3_parameters = {
                        "mu": initial_mu["3"],
                        "muhat": jnp.nan,
                        "pi": initial_pi["3"],
                        "pihat": jnp.nan,
                        "kappas": None,
                        "nu": jnp.nan,
                        "psis": None,
                        "omega": omega["3"],
                        "rho": rho["3"],
                        "node_type": "continuous",
                    }
                    x3: NodeType = x3_parameters, None, None

                    # Second level
                    x2_parameters = {
                        "mu": initial_mu["2"],
                        "muhat": jnp.nan,
                        "pi": initial_pi["2"],
                        "pihat": jnp.nan,
                        "kappas": (kappas["2"],),  # type: ignore
                        "nu": jnp.nan,
                        "psis": None,
                        "omega": omega["2"],
                        "rho": rho["2"],
                        "node_type": "continuous",
                    }
                    x2: NodeType = x2_parameters, None, (x3,)

                    # First level - Binary node
                    x1_parameters = {
                        "mu": initial_mu["1"],
                        "muhat": jnp.nan,
                        "pi": initial_pi["1"],
                        "pihat": jnp.nan,
                        "kappas": (kappas["1"],),
                        "nu": jnp.nan,
                        "psis": None,
                        "omega": omega["1"],
                        "rho": rho["1"],
                        "node_type": "binary",
                    }
                    x1: NodeType = x1_parameters, None, (x2,)

                    # Input node
                    input_node_parameters = {
                        "kappas": None,
                        "omega": omega_input,
                        "bias": self.bias,
                    }
                    self.input_node = input_node_parameters, x1, None

        elif model_type == "continuous":

            if self.n_levels == 2:

                if self.verbose:
                    print(
                        (
                            "Fitting the continuous Hierarchical Gaussian Filter (JAX) "
                            f"with {self.n_levels} levels."
                        )
                    )
                # Second level
                x2_parameters = {
                    "mu": initial_mu["2"],
                    "muhat": jnp.nan,
                    "pi": initial_pi["2"],
                    "pihat": jnp.nan,
                    "kappas": None,
                    "nu": jnp.nan,
                    "psis": None,
                    "omega": omega["2"],
                    "rho": rho["2"],
                }
                x2 = x2_parameters, None, None

            elif self.n_levels == 3:

                if self.verbose:
                    print(
                        (
                            "Fitting the continuous Hierarchical Gaussian Filter (JAX) "
                            f"with {self.n_levels} levels."
                        )
                    )

                # Third level
                x3_parameters = {
                    "mu": initial_mu["3"],
                    "muhat": jnp.nan,
                    "pi": initial_pi["3"],
                    "pihat": jnp.nan,
                    "kappas": None,
                    "nu": jnp.nan,
                    "psis": None,
                    "omega": omega["3"],
                    "rho": rho["3"],
                }
                x3 = x3_parameters, None, None

                # Second level
                x2_parameters = {
                    "mu": initial_mu["2"],
                    "muhat": jnp.nan,
                    "pi": initial_pi["2"],
                    "pihat": jnp.nan,
                    "kappas": (kappas["2"],),  # type: ignore
                    "nu": jnp.nan,
                    "psis": None,
                    "omega": omega["2"],
                    "rho": rho["2"],
                }
                x2: NodeType = x2_parameters, None, (x3,)  # type: ignore

            # First level
            x1_parameters = {
                "mu": initial_mu["1"],
                "muhat": jnp.nan,
                "pi": initial_pi["1"],
                "pihat": jnp.nan,
                "kappas": (kappas["1"],),
                "nu": jnp.nan,
                "psis": None,
                "omega": omega["1"],
                "rho": rho["1"],
            }
            x1 = x1_parameters, None, (x2,)

            # Input node
            input_node_parameters = {
                "kappas": None,
                "omega": omega_input,
                "bias": self.bias,
            }
            self.input_node = input_node_parameters, x1, None

        else:
            raise ValueError("model_type should be 'continuous' or 'binary'")

    def add_nodes(self, nodes: Tuple):
        """Add a custom node structure.
        Parameters
        ----------
        nodes : tuple
            The input node embeding the node hierarchy that will be updated during
            model fit.
        """
        node_validation(nodes, input_node=True)
        self.input_node = nodes  # type: ignore

    def input_data(
        self,
        input_data: np.ndarray,
    ):

        # Initialise the first values
        res_init = (
            self.input_node,
            {
                "time": input_data[0, 1],
                "value": input_data[0, 0] + self.bias,
                "surprise": jnp.array(0.0),
            },
        )

        # This is where the HGF functions are used to scan the input time series
        last, final = scan(loop_inputs, res_init, input_data[1:, :])

        # Save results in the HGF dictionary
        self.hgf_results = {}
        self.hgf_results["last"] = last  # The last tuple returned
        self.hgf_results[
            "final"
        ] = final  # The commulative update of the nodes and results
        self.hgf_results["data"] = input_data  # type: ignore

        return self

    def plot_trajectories(self, **kwargs):
        plot_trajectories(hgf=self, **kwargs)

    def plot_correlations(self):
        plot_correlations(hgf=self)

    def surprise(
        self,
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Tuple = (),
    ):
        """Returns the model surprise (negative log probability) given the input data
        and a response function and additional parameters to the response function.

        Parameters
        ----------
        response_function : callable
            The response function to use to compute the model surprise.
        response_function_parameters : tuple
            (Optional) Additional parameters to the response function.

        Returns
        -------
        surprise : float
            The model surprise given the input data and the response function.

        """

        return response_function(
            hgf_results=self.hgf_results,
            response_function_parameters=response_function_parameters,
        )
