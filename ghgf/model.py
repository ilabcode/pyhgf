# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.lax import scan

from ghgf.jax import loop_inputs, node_validation
from ghgf.plots import plot_trajectories


class HGF(object):
    """The standard 2 or 3 levels HGF for continuous inputs.

    Attributes
    ----------
    verbose : bool
        Verbosity level.
    n_levels : int
        The number of hierarchies in the model. Cannot be less than 2.
    model_type : str
        The model implemented (can be `"AR1"` or `"GRW"`).
    nodes : tuple
        The nodes hierarchy.

    Notes
    -----
    The model used by the perceptual model is defined by the `model_type` parameter
    (can be `"GRW"` or `"AR1"`). If `model_type` is not provided, the class will
    try to determine it automatically by looking at the `rho` and `phi` parameters.
    If `rho` is provided `model_type="GRW"`, if `phi` is provided
    `model_type="AR1"`. If both `phi` and `rho` are `None` an error will be
    returned.

    Examples
    --------

    """

    def __init__(
        self,
        n_levels: Optional[int] = 2,
        model_type: str = "continuous",
        initial_mu: Dict[str, DeviceArray] = {"1": jnp.array(0.0), "2": jnp.array(0.0)},
        initial_pi: Dict[str, DeviceArray] = {"1": jnp.array(1.0), "2": jnp.array(1.0)},
        omega_input: DeviceArray = jnp.log(1e-4),
        omega: Dict[str, DeviceArray] = {"1": jnp.array(-10.0), "2": jnp.array(-10.0)},
        kappas: Dict[str, DeviceArray] = {"1": jnp.array(1.0)},
        rho: Dict[str, DeviceArray] = {"1": jnp.array(0.0), "2": jnp.array(0.0)},
        bias: DeviceArray = jnp.array(0.0),
        verbose: bool = True,
    ):

        """Parameterization of the HGF model.

        Parameters
        ----------
        n_levels : int | None
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Default sets to `2`.
        model_type : str
            The model type to use (can be "continuous" or "binary").
        initial_mu : dict
            Dictionnary containing the initial values for the `initial_mu` parameter at
            different levels of the hierarchy. Defaults set to `{"1": 0.0, "2": 0.0}`
            for a 2-levels model.
        initial_pi : dict
            Dictionnary containing the initial values for the `initial_pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults set to `{"1": 1.0, "2": 1.0}` for
            a 2-levels model.
        omega : dict
            Dictionnary containing the initial values for the `omega` parameter at
            different levels of the hierarchy. Omegas represent the tonic part of the
            variance (the part that is not affected by the parent node). Defaults set to
            `{"1": -10.0, "2": -10.0}` for a 2-levels model. This parameters only when
            `model_type="GRW"`.
        omega_input : float
            Default value sets to `np.log(1e-4)`. Represents the noise associated with
            the input.
        rho : dict
            Dictionnary containing the initial values for the `rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random
            walk. Only required when `model_type="GRW"`. Defaults set all entries to
            `0` according to the number of required levels.
        kappas : dict
            Dictionnary containing the initial values for the `kappa` parameter at
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

        """
        if model_type == "binary":
            raise NotImplementedError

        self.model_type = model_type
        self.verbose = verbose
        self.n_levels = n_levels
        self.bias = bias

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
                "kappas": (kappas["2"],),
                "nu": jnp.nan,
                "psis": None,
                "omega": omega["2"],
                "rho": rho["2"],
            }
            x2 = x2_parameters, None, (x3,)  # type: ignore

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
        input_data,
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

        # Save results in the HGF instance
        self.last = last  # The last tuple returned
        self.final = final  # The commulative update of the nodes and results
        self.data = input_data  # The input data

    def plot_trajectories(self, backend: str = "matplotlib", **kwargs):
        plot_trajectories(model=self, backend=backend, **kwargs)

    def surprise(self):

        _, results = self.final
        return jnp.sum(results["surprise"])
