# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional, Tuple

import jax.numpy as jnp
from jax.interpreters.xla import DeviceArray
from jax.lax import scan
from numpyro.distributions import Distribution, constraints

from ghgf.hgf_jax import loop_inputs, node_validation


class HGF(object):
    """Generic HGF model"""

    def __init__(
        self,
        n_levels: Optional[int] = None,
        model_type: Optional[str] = None,
        initial_mu: Dict[str, float] = {"1": jnp.array(0.0), "2": jnp.array(0.0)},
        initial_pi: Dict[str, float] = {"1": jnp.array(1.0), "2": jnp.array(1.0)},
        omega_input: float = jnp.log(1e-4),
        omega: Dict[str, float] = {"1": -10.0, "2": -10.0},
        kappa: Dict[str, float] = {"1": jnp.array(1.0)},
        rho: Optional[Dict[str, float]] = None,
        phi: Optional[Dict[str, float]] = None,
        m: Dict[str, float] = None,
        verbose: bool = True,
    ):

        """The standard n-level HGF for continuous inputs with JAX backend.

        The standard continuous HGF can implements the Gaussian Random Walk and AR1
        perceptual models.

        Parameters
        ----------
        n_levels : int | None
            The number of hierarchies in the perceptual model (can be `1` or `2`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Default sets to `None`.
        model_type : str or None
            The model type to use (can be "GRW" or "AR1"). If `model_type` is not
            provided, it is infered from the parameters provided. If both `phi` and
            `rho` are None or dictionnary, an error is returned.
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
            Default value sets to `np.log(1e-4)`. Represent the noise associated with
            the input.
        rho : dict | None
            Dictionnary containing the initial values for the `rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random
            walk. Only required when `model_type="GRW"`. Defaults set all entries to
            `0` according to the number of required levels.
        kappa : dict
            Dictionnary containing the initial values for the `kappa` parameter at
            different levels of the hierarchy. Kappa represents the phasic part of the
            variance (the part that is affected by the parents nodes) and will defines
            the strenght of the connection between the node and the parent node. Often
            fixed to 1. Defaults set to `{"1": 1.0}` for a 2-levels model. Only
            required when `model_type="GRW"`.
        phi : dict | None
            Dictionnary containing the initial values for the `phi` parameter at
            different levels of the hierarchy. Phi should always be between 0 and 1.
            Defaults set all entries to `0` according to the number of required levels.
            `phi` is only required when `model_type="AR1"`.
        m : dict or None
            Dictionnary containing the initial values for the `m` parameter at
            different levels of the hierarchy. Defaults set all entries to `0`
            according to the number of required levels. `m` is only required when
            `model_type="AR1"`.
        verbose : bool
            Default is `True` (show bar progress).

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

        self.verbose = verbose
        self.n_levels = n_levels
        if self.verbose:
            print(
                (
                    "Fitting the continuous Hierarchical Gaussian Filter (JAX backend) "
                    f"with {self.n_levels} levels."
                )
            )

        if n_levels == 2:

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

            # First level
            x1_parameters = {
                "mu": initial_mu["1"],
                "muhat": jnp.nan,
                "pi": initial_pi["1"],
                "pihat": jnp.nan,
                "kappas": (kappa["1"],),
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
        self.input_node = nodes

    def input_data(
        self,
        input_data,
        time: DeviceArray = jnp.array(0.0),
        value: DeviceArray = jnp.nan,
        surprise: DeviceArray = jnp.array(0.0),
    ):

        # Initialise the first values
        res_init = (
            self.input_node,
            {
                "time": time,
                "value": value,
                "surprise": surprise,
            },
        )

        # This is where the HGF functions are used to scan the input time series
        last, final = scan(loop_inputs, res_init, input_data)

        # Store ouptut values
        self.last = last  # The last tuple returned
        self.final = final  # The commulative update of the nodes and results

    def surprise(self):

        _, results = self.final
        surprise = jnp.sum(results["surprise"])
        return jnp.where(jnp.isnan(surprise), -jnp.inf, surprise)


class HGFDistribution(Distribution):

    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        input_data,
        model_type="GRW",
        omega_1=None,
        omega_2=None,
        rho_1=None,
        rho_2=None,
        pi_1=None,
        pi_2=None,
        mu_1=None,
        mu_2=None,
        kappa=jnp.array(1.0),
    ):
        self.input_data = input_data
        self.model_type = model_type
        self.omega_1 = omega_1
        self.omega_2 = omega_2
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.pi_1 = pi_1
        self.pi_2 = pi_2
        self.mu_1 = mu_1
        self.mu_2 = mu_2
        self.kappa = kappa
        super().__init__(batch_shape=(1,), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError

    def log_prob(self, value):

        hgf = HGF(
            n_levels=2,
            model_type="GRW",
            initial_mu={"1": self.mu_1, "2": self.mu_2},
            initial_pi={"1": self.pi_1, "2": self.pi_2},
            omega={"1": self.omega_1, "2": self.omega_2},
            rho={"1": self.rho_1, "2": self.rho_2},
            kappa={"1": self.kappa},
            verbose=False,
        )

        # Create the input structure
        res_init = (
            hgf.input_node,
            {
                "time": jnp.array(0.0),
                "value": jnp.array(0.0),
                "surprise": jnp.array(0.0),
            },
        )

        # This is where the HGF functions are used to scan the input time series
        _, final = scan(loop_inputs, res_init, self.input_data)
        nodes, results = final
        self.nodes = nodes
        self.final = final
        self.results = results

        surprise = jnp.sum(results["surprise"])

        # Return the negative surprise or -Inf if the model cannot fit
        return jnp.where(jnp.isnan(surprise), -jnp.inf, -surprise)
