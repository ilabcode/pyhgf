# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.interpreters.xla import DeviceArray
from jax.scipy.stats import norm

if TYPE_CHECKING:
    from ghgf.model import HGF


class GaussianObs:
    """The gaussian observation response function.

    The probability of the observed response is given by:
        p(y | mu_hat_1) ~ N(mu_hat_1, zeta > 0)

    Where mu_hat_1 is the mean of the underlying normal distribution, and zeta is the
    variance of the underlying normal distribution.

    """

    def __init__(self):
        if "zeta" not in self.parameters:
            raise ValueError("The parameter dictionnary should contain a zeta variable")

    def fit(self):
        """Fit the responses and return the negative log probability.

        Parameters
        ----------
        responses : list
            The responses provided by the participant.

        Returns
        -------
        neg_log_p : floats
            The sum of the negative log probability.

        """

        sigma = np.sqrt(self.parameters["zeta"])
        mu_hat_1 = self.model.x1.mus[1:]

        return (-np.log(sigma) - ((self.responses - mu_hat_1) / sigma) ** 2).sum()


class HRD:
    """The binary response function for the HRD task.

    Parameters
    ----------
    model : py:class:`ghgf.hgf.model.HGF` instance
        The HGF model (JAX backend).
    final : tuple
        Final node structure after perceptual model fit.
    time : DeviceArray
        The time vector.
    triggers_idx : DeviceArray
        The triggers indexs (sample).
    tones : DeciceArray
        The frequencies of the tones presented at each trial.
    decisions : DeviceArray
        The participant decision (boolean) where `1` stands for Faster and `0` for
        Slower.
    sigma_tone : DeviceArray
        The precision of the tone perception. Defaults to `1`.

    """

    def __init__(
        self,
        model: "HGF",
        final: Tuple,
        time: DeviceArray,
        triggers_idx: DeviceArray,
        tones: DeviceArray,
        decisions: DeviceArray,
        sigma_tone: DeviceArray = jnp.array([1.0]),
    ):

        assert len(tones) == len(triggers_idx)

        # The continuous HGF model
        self.model = model

        # The time vector
        self.time = time

        # The triggers indexs
        self.triggers_idx = triggers_idx

        # The tone frequency at each trial
        self.tones = tones

        # The participant's decisions
        self.decisions = decisions

        # The standard deviation for the tone precision
        self.sigma_tone = sigma_tone

        # The estimated heart rate (mu_1)
        self.mu_1 = final[0][1][0]["mu"]

        # The precision of the first level estimate
        self.pi_1 = final[0][1][2][0][0]["pi"]

    def surprise(self) -> DeviceArray:
        """Fit the responses and return the negative log probability.

        Parameters
        ----------


        Returns
        -------
        surprise : DeviceArray
            The model surprise given the responses.

        """

        # Extract the values of mu_1 and pi_1 for each trial
        # --------------------------------------------------

        # First define a function to extract values for one trigger
        @jit
        def extract(trigger: int, mu_1=self.mu_1, pi_1=self.pi_1, time=self.time):
            return (
                mu_1[jnp.sum(time <= trigger) - 1],
                pi_1[jnp.sum(time <= trigger) - 1],
            )

        hr, precision = vmap(extract)(self.triggers_idx)
        hr = 60000 / jnp.exp(hr)  # Convert the values to BPM

        # The probabilit of answering Faster
        cdf = norm.cdf(
            0,
            loc=hr - self.tones,
            scale=jnp.sqrt(1 / precision + (self.sigma_tone ** 2)),
        )

        # The surprise (sum of the -log(p)) of the model given the answers
        surprise = jnp.where(self.decisions, -jnp.log(cdf), -jnp.log(1 - cdf)).sum()

        return surprise
