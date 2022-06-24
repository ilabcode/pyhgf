# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from functools import partial
from typing import Callable, Dict, Optional

import jax.numpy as jnp
from jax import jit
from jax.interpreters.xla import DeviceArray
from numpyro.distributions import Distribution, constraints

from ghgf.model import HGF
from ghgf.response import gaussian_surprise


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

        surprise = (
            HGF(
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
            .input_data(data)
            .surprise(
                response_function=self.response_function,
                response_function_parameters=self.response_function_parameters,
            )
        )

        # Return the negative of the sum of the log probabilities
        return -surprise
