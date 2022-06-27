# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from functools import partial
from math import log
from typing import Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.interpreters.xla import DeviceArray
from numpyro.distributions import Distribution, constraints

from ghgf.model import HGF
from ghgf.response import gaussian_surprise


class HGFDistribution(Distribution):
    """The HGF distribution. Should be used in the context of the declaration of a
    Numpyro model.

    Examples
    --------
    This class should be used in the context of a Numpyro probabilistic model to
    estimate one or many parameter-s probability density with MCMC sampling.

    .. python::

        from ghgf import load_data
        from ghgf.numpyro import HGFDistribution
        from numpyro.infer import MCMC, NUTS
        import numpyro as npy
        from jax import random
        import numpyro.distributions as dist
        import numpy as np
        import arviz as az

        # Load and format the input data (2d array with values and time vector)
        timeseries = load_data("continuous")
        input_data = np.array(
            [timeseries, np.arange(1, len(timeseries) + 1, dtype=float)]
        ).T

        # Create the probabilistic model
        # Here we are fixing all the paramters to arbitrary values and sampling the
        # posterior probability density of Omega in the second level, using a normal
        # distribution as prior : ω_2 ~ N(-11, 2).

        def model(input_data):

            ω_1 = 0.0
            ω_2 = npy.sample("ω_2", dist.Normal(-11.0, 2.0))
            rho_1 = 0.0
            rho_2 = 0.0
            pi_1 = 1e4
            pi_2 = 1e1
            μ_1 = input_data[0, 0]
            μ_2 = 0.0
            kappa_1 = 1.0

            npy.sample(
                "hgf_log_prob",
                HGFDistribution(
                    input_data=input_data,
                    omega_1=ω_1,
                    omega_2=ω_2,
                    rho_1=rho_1,
                    rho_2=rho_2,
                    pi_1=pi_1,
                    pi_2=pi_2,
                    mu_1=μ_1,
                    mu_2=μ_2,
                    kappa_1=kappa_1,
                ),
            )

        # Sample the model using Numpyro NUST sampler with 1000 warmup and 1000 samples
        rng_key = random.PRNGKey(0)
        rng_key, rng_key_ = random.split(rng_key)

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
        mcmc.run(rng_key_, input_data=input_data)

        # Print a summary of the results using Arviz
        numpyro_samples = az.from_numpyro(mcmc)
        az.summary(numpyro_samples, var_names="ω_2")
        ===  =======  =====  =======  =======  =====  =====  ===  ===  ===
        ω_2  -12.948  1.363  -15.405  -10.596  0.048  0.034  854  781  nan
        ===  =======  =====  =======  =======  =====  =====  ===  ===  ===

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
        input_data: np.ndarray,
        omega_1: float,
        omega_2: float,
        rho_1: float,
        rho_2: float,
        pi_1: float,
        pi_2: float,
        mu_1: float,
        mu_2: float,
        omega_input: float = log(1e-4),
        omega_3: Optional[float] = None,
        rho_3: Optional[float] = None,
        pi_3: Optional[float] = None,
        mu_3: Optional[float] = None,
        kappa_1: float = 1.0,
        kappa_2: Optional[float] = None,
        bias: DeviceArray = 0.0,
        response_function: Callable = gaussian_surprise,
        response_function_parameters: Dict = {},
        n_levels: int = 2,
        model_type: str = "continuous",
    ):
        """
        Parameters
        ----------
        input_data : np.ndarray
            A n x 2 DeviceArray (:, (Data, Time)) where n is the number of samples in
            the time series. The first column should contain the values and the second
            column should contain the time vector.
        model_type : str
            The model type to use (can be "continuous" or "binary").
        n_levels : int | None
            The number of hierarchies in the perceptual model (can be `2` or `3`). If
            `None`, the nodes hierarchy is not created and might be provided afterward
            using `add_nodes()`. Default sets to `2`.
        omega_1, omega_2, omega_3, rho_1, rho_2, rho_3, pi_1, pi_2, pi_3, mu_1, mu_2,
        mu_3, kappa_1, kappa_2, bias : DeviceArray
            The HGD parameters (see py:class:`ghgf.model.HGF` for details).
        response_function : callable
            The response function to use to compute the model surprise.
        response_function_parameters : tuple
            (Optional) Additional parameters to the response function.

        See also
        --------
        py:func:`ghgf.model.HGF`

        """
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

        # Return the sum of the log probabilities
        return -surprise
