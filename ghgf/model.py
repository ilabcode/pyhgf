# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
from jax import jit
from jax.interpreters.xla import DeviceArray
from jax.lax import scan
from numpyro.distributions import Distribution, constraints

from ghgf.hgf_jax import loop_inputs, node_validation
from ghgf.plots import plot_trajectories
from ghgf.response import gaussian_surprise


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


class HGFDistribution(Distribution):
    """
    Parameters
    ----------
    input_data : DeviceArray
        A n x 2 DeviceArray (:, (Data, Time)).
    response_function : str
        Name of the response function to use to compute model surprise when provided
        evidences. Defaults to `"GaussianSurprise"` (continuous inputs).

    """

    support = constraints.real
    has_rsample = False
    arg_constraints = {
        "omega_1": constraints.real,
        "omega_2": constraints.real,
        "omega_3": constraints.real,
        "omega_input": constraints.real,
        "rho_1": constraints.real,
        "rho_2": constraints.real,
        "rho_3": constraints.real,
        "pi_1": constraints.positive,
        "pi_2": constraints.positive,
        "pi_3": constraints.positive,
        "mu_1": constraints.real,
        "mu_2": constraints.real,
        "mu_3": constraints.real,
        "kappa_1": constraints.unit_interval,
        "kappa_2": constraints.unit_interval,
        "bias": constraints.real,
    }
    reparametrized_params = [
        "omega_1",
        "omega_2",
        "omega_3",
        "omega_input",
        "rho_1",
        "rho_2",
        "rho_3",
        "pi_1",
        "pi_2",
        "pi_3",
        "mu_1",
        "mu_2",
        "mu_3",
        "kappa_1",
        "kappa_2",
        "bias",
    ]

    def __init__(
        self,
        input_data: DeviceArray,
        model_type: str = "GRW",
        n_levels: int = 2,
        omega_1: Optional[DeviceArray] = None,
        omega_2: Optional[DeviceArray] = None,
        omega_3: Optional[DeviceArray] = None,
        omega_input: DeviceArray = jnp.log(1e-4),
        rho_1: Optional[DeviceArray] = None,
        rho_2: Optional[DeviceArray] = None,
        rho_3: Optional[DeviceArray] = None,
        pi_1: Optional[DeviceArray] = None,
        pi_2: Optional[DeviceArray] = None,
        pi_3: Optional[DeviceArray] = None,
        mu_1: Optional[DeviceArray] = None,
        mu_2: Optional[DeviceArray] = None,
        mu_3: Optional[DeviceArray] = None,
        kappa_1: DeviceArray = jnp.array(1.0),
        kappa_2: DeviceArray = jnp.array(1.0),
        bias: DeviceArray = jnp.array(0.0),
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Dict = {},
    ):
        self.input_data = input_data
        self.model_type = model_type
        self.n_levels = n_levels
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.omega_3 = omega_3
        self.omega_input = omega_input
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.rho_3 = rho_3
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.pi_3 = pi_3
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.mu_3 = mu_3
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.response_function = response_function
        self.response_function_parameters = response_function_parameters
        self.bias = bias
        super().__init__(batch_shape=(1,), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def log_prob(self, value) -> jnp.DeviceArray:
        """Compute the log probability from the HGF model given the data and
        parameters."""
        data = self.input_data

        # Format HGF parameters
        initial_mu = {"1": self.mu_1, "2": self.mu_2}
        initial_pi = {"1": self.pi_1, "2": self.pi_2}
        omega = {"1": self.omega_1, "2": self.omega_2}
        rho = {"1": self.rho_1, "2": self.rho_2}
        kappas = {"1": self.kappa_1}

        # Add parameters for a third level if required
        initial_mu["3"] = self.mu_3 if self.n_levels == 3 else None
        initial_pi["3"] = self.pi_3 if self.n_levels == 3 else None
        omega["3"] = self.omega_3 if self.n_levels == 3 else None
        rho["3"] = self.rho_3 if self.n_levels == 3 else None
        kappas["2"] = self.kappa_2 if self.n_levels == 3 else None

        hgf = HGF(
            n_levels=self.n_levels,
            model_type=self.model_type,
            initial_mu=initial_mu,
            initial_pi=initial_pi,
            omega=omega,
            omega_input=self.omega_input,
            rho=rho,
            kappas=kappas,
            bias=self.bias,
            verbose=False,
        )

        # Create the input structure - use the first row from the input data
        # This is where the HGF functions is used to scan the input time series
        hgf.input_data(data)

        # Return the model evidence
        logp = self.response_function(
            hgf=hgf, response_function_parameters=self.response_function_parameters
        )

        # Return the negative of the sum of the log probabilities
        return -logp
