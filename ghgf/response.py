# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from jax.interpreters.xla import DeviceArray
from jax.lax import dynamic_slice
from jax.scipy.stats import norm


def gaussian_surprise(hgf_results: Dict, response_function_parameters):
    """The sum of the gaussian surprise along the time series (continuous HGF).

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results.
    response_function_parameters : None
        No additional parameters are required.

    Return
    ------
    surprise : DeviceArray
        The model surprise given the input data.

    """
    _, results = hgf_results["final"]

    # Fill surprises with zeros if invalid input
    this_surprise = jnp.where(
        jnp.any(jnp.isnan(hgf_results["data"][1:, :]), axis=1), 0.0, results["surprise"]
    )

    # Return an infinite surprise if the model cannot fit
    this_surprise = jnp.where(jnp.isnan(this_surprise), jnp.inf, this_surprise)

    # Sum the surprise for this model
    surprise = jnp.sum(this_surprise)

    return surprise


def hrd_behaviors(
    hgf_results: Dict,
    response_function_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> DeviceArray:
    """The binary response function for the HRD task.

    The participant try to correctly decide if the tone currently presented
    (`sound_frequency`) is faster or slower than her/his heart rate, using the belief
    about this values `heart_rate_belief`. `heart_rate_belief` is the average of
    heart rate beliefs (the first level of the HGF model, `mu_1`) over 5 seconds (which
    corresponds to the listening period during the HRD task). The noise around this
    perception is represented by the precision of this parameter (`pi_1`) averaged over
    the same 5 seconds. The probability of answering slower (`cdf`) is then generated
    from a cumulative normal distribution where `loc = heart_rate_belief -
    sound_frequency` and `scale = jnp.sqrt(1 / precision)`. The surprise of the model is
    given by `surprise = -jnp.log(cdf)` if the participant effectively answered
    `"Slower"`, and `surprise = -jnp.log(1 - cdf)`. These values are then averaged over
    all trials to get the final surprise.

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results, which include the updated node structure
        and the trajectories for the different levels of the model.
    response_function_parameters : tuple
        Additional parameters used to compute the log probability from the HGF model.
        Here, it contains information about the HRD task for a given particpant:

        - triggers_idx : np.ndarray
            The triggers indexs (in samples, at 1kHz).
        - sound_frequency : np.ndarray
            The tones frequencies that were presented at each trial, converted from BPM
            to log(RR intervals) (ms) to match with the units of the input data.
        - decisions : np.ndarray
            The participant decision (boolean) where `1` stands for Slower and `0` for
            Faster.

    Return
    ------
    surprise : DeviceArray
        The model evidence given the participant's behavior.

    """

    # Unpack the additional parameters
    (
        triggers_idx,
        sound_frequency,
        decisions,
        time_vector,
    ) = response_function_parameters

    # The trajectories of the heart rate beliefs (first level of the HGF, mu_1)
    mu_1 = hgf_results["final"][0][1][0]["mu"]

    # The trajectories of the precision of the heart rate beliefs
    # (first level of the HGF, pi_1)
    pi_1 = hgf_results["final"][0][1][0]["pi"]

    # The model surprise along the entire time series - This is used to return an
    # infinite surprise in case the model cannot fit at all given the parameters
    model_surprise = hgf_results["final"][1]["surprise"]

    # The time vector - this is used to interpolate the trajectories
    time = hgf_results["data"][:, 1] * 1000

    # Interpolate the model trajectories (mu_1 and pi_1) at 1000 Hz - The continuous
    # HGF use and unequally spaced time vector as input, corresponding to the detection
    # of a heartbeat. We need to interpolate this vector to e.g. 1kHz, so we can
    # average those values over 5 seconds after each triggers.
    new_mu1 = jnp.interp(
        x=time_vector,
        xp=time[1:],
        fp=mu_1,
    )
    new_pi1 = jnp.interp(
        x=time_vector,
        xp=time[1:],
        fp=pi_1,
    )

    # This part just extract the average of mu_1 and pi_1 over 5 seconds after the
    # triggers. It has to be done this way to let JAX use JIT compilation, but this is
    # just a dynamic array indexing.
    # ----------------------------------------------------------------------------------
    # First define a function to extract values for one trigger using dynamic_slice()
    def extract(
        trigger: np.ndarray,
        new_mu1: DeviceArray = new_mu1,
        new_pi1: DeviceArray = new_pi1,
    ):
        return (
            dynamic_slice(new_mu1, (trigger,), (5000,)).mean(),
            dynamic_slice(new_pi1, (trigger,), (5000,)).mean(),
        )

    # Map this function along the triggers dimension using vmap()
    heart_rate_belief, precision = vmap(extract)(triggers_idx)
    # ----------------------------------------------------------------------------------

    # Generate the probabilities of answering Slower given the heart_rate_belief and
    # the precision as parameters of a cumulative normal distribution. We substract
    # sound_frequency and estimate the probability as 0.
    cdf = norm.cdf(
        0,
        loc=heart_rate_belief - sound_frequency,
        scale=jnp.sqrt(1 / precision),
    )

    # Force probabilities to be greater or smaller than 0 or 1 - This is to deal with
    # cases where the model has extrem probabilities of a "slower" responses.
    cdf = jnp.where(cdf == 1.0, 1 - 1e-6, cdf)
    cdf = jnp.where(cdf == 0.0, 1e-6, cdf)

    # The surprise (sum of the log(p)) of the model given the answers
    surprise = jnp.where(decisions, -jnp.log(cdf), -jnp.log(1 - cdf)).sum()

    # Return an infinite surprise if the model could not fit in the first place
    # This will then overwrite the previous surprise value
    surprise = jnp.where(jnp.any(jnp.isnan(model_surprise)), jnp.inf, surprise)

    return surprise


def hrd_ideal_oberver(
    hgf_results: Dict,
    response_function_parameters: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> DeviceArray:
    """The binary response function for the HRD task given by an ideal observer.

    Parameters
    ----------
    hgf_results : dict
        Dictionary containing the HGF results.
    response_function_parameters : tuple
        Additional parameters used to compute the log probability from the HGF model:

       - triggers_idx : np.ndarray
            The triggers indexs (sample).
        - tones : np.ndarray
            The frequencies of the tones presented at each trial.
        - decisions : np.ndarray
            The participant decision (boolean) where `1` stands for Slower and `0` for
            Faster. The coding is different from what is used by the Psi staircase as
            the input use different units (log(RR), in miliseconds).

    Return
    ------
    surprise : DeviceArray
        The model evidence given the participant's behavior.

    """
    (
        triggers_idx,
        tones,
        condition,
    ) = response_function_parameters

    # The estimated heart rate (mu_1)
    mu_1 = hgf_results["final"][0][1][0]["mu"]

    # The surprise along the entire time series
    model_surprise = hgf_results["final"][1]["surprise"]

    # The time vector
    time = hgf_results["data"][:, 1]

    # Extract the values of mu_1 and pi_1 for each trial
    # The heart rate belief and the precision of the heart rate belief
    # --------------------------------------------------

    # First define a function to extract values for one trigger
    @jit
    def extract(trigger: int, mu_1=mu_1, time=time):
        idx = jnp.sum(time <= trigger) + 1
        return dynamic_slice(mu_1, (idx,), (3,)).mean()

    hr = vmap(extract)(triggers_idx)

    # The probability of answering Slower
    cdf = norm.cdf(
        0,
        loc=hr - tones,
        scale=1e-6,
    )

    # Force probabilities to be greater or smaller than 0 or 1
    cdf = jnp.where(cdf == 1.0, 1 - 1e-6, cdf)
    cdf = jnp.where(cdf == 0.0, 1e-6, cdf)

    # The surprise (negative sum of the log(p)) of the model given the answers
    surprise = jnp.where(condition, -jnp.log(cdf), -jnp.log(1 - cdf)).sum()

    # Return an infinite surprise if the model could not fit in the first place
    surprise = jnp.where(jnp.any(jnp.isnan(model_surprise)), jnp.inf, surprise)

    return surprise
