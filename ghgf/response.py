from typing import List, Union

import numpy as np
from jax.scipy.stats import norm

from ghgf.hgf import HgfUpdateError, Model


class ResponseFunction:
    """Response function objects.

    Parameters
    ----------
    model : py:class:`ghgf.hgf.Model` instance
        The perceptual model.
    parameters : list
        The parameters of the response function that should be optimized.
    responses : list or np.ndarray
        The responses provided by the participant. Can be a list or a Numpy array with
        n >= 1 dimensions where axis 0 encodes the number of trials.

    """

    def __inti__(
        self, model: Model, parameters: List, responses: Union[List, np.ndarray]
    ):

        # Check that responses and perceptual model fit together
        if isinstance(responses, list):
            if len(responses) != len(model.x2.mus[1:]):
                raise ValueError(
                    (
                        "The responses provided does not match the lenght of the "
                        "beliefs in the perceptual model."
                    )
                )
            self.responses = np.asarray(responses)
        elif isinstance(responses, np.ndarray):
            if responses.shape[0] != len(model.x2.mus[1:]):
                raise ValueError(
                    (
                        "The first dimension of the  responses provided does not match "
                        "the lenght of the beliefs in the perceptual model."
                    )
                )
            self.responses = responses

        self.model = model
        self.parameters = parameters


class GaussianObs(ResponseFunction):
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


class HRD(ResponseFunction):
    """The gaussian observation response function.

    The probability of the observed response is given by:
        p(y | mu_hat_1) ~ N(mu_hat_1, zeta > 0)

    Where mu_hat_1 is the mean of the underlying normal distribution, and zeta is the
    variance of the underlying normal distribution.

    """

    def __init__(self):

        for param in ["sigma_tone", "extero_tone"]:
            if param not in self.parameters:
                raise ValueError(
                    f"The parameter dictionnary should contain a {param} variable"
                )
        self.parameters["sigma_tone"] = 1  # Fix this parameter for now
        if len(self.parameters["extero_tone"]) != len(self.reponses):
            raise ValueError(
                "The length of responses and exteroceptives tones are inconsistents"
            )

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
        mu_hat_1 = self.model.x1.mus[1:]  # The heart rate estimate at trial t

        return -np.log(
            norm.cdf(
                0,
                loc=mu_hat_1 - self.parameters["extero_tone"],
                scale=np.sqrt(
                    1 / self.model.x1.pis[1:] + (self.parameters["sigma_tone"] ** 2)
                ),
            )
        )


class ResponseModel:
    """Response model

    Parameters
    ----------
    model : py:class:`ghgf.hgf.Model` instance
        The perceptual model.
    parameters : list
        The parameters of the response model.
    likelihood : py:class:`ghgf.response.ResponseFunction`
        The likelihood of the response model.

    """

    def __init__(
        self,
        model: Model,
        parameters: List,
        likelihood: ResponseFunction,
    ):

        self.model = model
        self.parameters = parameters
        self.likelihood = likelihood

    def objective_funtion(self):
        """Optimization of the parameters of the perceptual and response model.

        Returns
        -------
        f : Callable
            The objective function that have to be optimized.
        """
        nperc = len  # Delimiter for retrieving perceptual parameters

        def f(self, parameters: Union[List, np.array], nperc: int = nperc):

            ##################################################
            # Reinitialize the HGF filter using new parameters
            ##################################################
            trans_values_backup = self.var_param_trans_values
            self.var_param_trans_values = parameters[:nperc]
            try:
                self.recalculate()
                return -self.log_joint()
            except HgfUpdateError:
                self.var_param_trans_values = trans_values_backup
                return np.inf

            ###############################################
            # Generate response and compute log-probability
            ###############################################
            logp = self.likelihood(
                responses=self.responses,
                model=self.model,
                parameters=parameters[nperc:],
            ).fit()

            return logp.sum()
