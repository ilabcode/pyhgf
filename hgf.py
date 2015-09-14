# hgf.py
"""The HGF time series model."""

import numpy as np
import warnings


# Turn warnings into exceptions
warnings.simplefilter('error')


# HGF continuous state node
class StateNode(object):
    """HGF continuous state node"""
    def __init__(self,
                 *,
                 initial_mu,
                 initial_pi,
                 rho=0.0,
                 phi=0.0,
                 m=0.0,
                 omega=0.0):

        # Sanity check
        if rho and phi:
            raise NodeConfigurationError(
                'hgf.StateNode: rho (drift) and phi (AR(1) parameter) may ' +
                'not be non-zero at the same time.')

        # Initialize parameter attributes
        self.initial_mu = initial_mu
        self.initial_pi = initial_pi
        self.rho = rho
        self.phi = phi
        self.m = m
        self.omega = omega
        self.psis = []
        self.kappas = []

        # Initialize parents
        self.va_pas = []
        self.vo_pas = []

        # Initialize time series
        self.reset()

    def reset(self):
        self.times = [0.0]
        self.pihats = [None]
        self.pis = [self.initial_pi]
        self.muhats = [None]
        self.mus = [self.initial_mu]
        self.nus = [None]

    def add_value_parent(self, *, parent, psi):
        self.va_pas.append(parent)
        self.psis.append(psi)

    def add_volatility_parent(self, *, parent, kappa):
        self.vo_pas.append(parent)
        self.kappas.append(kappa)

    def new_muhat(self, time):
        t = time - self.times[-1]
        driftrate = self.rho
        for i, va_pa in enumerate(self.va_pas):
            driftrate += self.psis[i] * self.va_pas[i].mus[-1]
        return self.mus[-1] + t * driftrate

    def _new_nu(self, time):
        t = time - self.times[-1]
        logvol = self.omega
        for i, vo_pa in enumerate(self.vo_pas):
            logvol += self.kappas[i] * self.vo_pas[i].mus[-1]
        return t * np.exp(logvol)

    def new_pihat_nu(self, time):
        new_nu = self._new_nu(time)
        return [1 / (1 / self.pis[-1] + new_nu), new_nu]

    def vape(self):
        return self.mus[-1] - self.muhats[-1]

    def vope(self):
        return ((1 / self.pis[-1] + self.vape()**2) *
                self.pihats[-1] - 1)

    def update_parents(self, time):
        va_pas = self.va_pas
        vo_pas = self.vo_pas

        if not va_pas and not vo_pas:
            return

        pihat = self.pihats[-1]

        # Update value parents
        psis = self.psis
        vape = self.vape()

        for i, va_pa in enumerate(va_pas):
            pihat_pa, nu_pa = va_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + psis[i]**2 * pihat

            muhat_pa = va_pa.new_muhat(time)
            mu_pa = muhat_pa + psis[i] * pihat / pi_pa * vape

            va_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

        # Update volatility parents
        nu = self.nus[-1]
        kappas = self.kappas
        vope = self.vope()

        for i, vo_pa in enumerate(vo_pas):
            pihat_pa, nu_pa = vo_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + 0.5 * (kappas[i] * nu * pihat)**2 * \
                (1 + (1 - 1 / (nu * self.pis[-2])) * vope)
            if pi_pa <= 0:
                raise NegativePosteriorPrecisionError(
                    'Parameters values are in region where model ' +
                    'assumptions are violated.')

            muhat_pa = vo_pa.new_muhat(time)
            mu_pa = muhat_pa + 0.5 * kappas[i] * nu * pihat / pi_pa * vope

            vo_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

    def update(self, time, pihat, pi, muhat, mu, nu):
        self.times.append(time)
        self.pihats.append(pihat)
        self.pis.append(pi),
        self.muhats.append(muhat)
        self.mus.append(mu)
        self.nus.append(nu)

        self.update_parents(time)


# HGF binary state node
class BinaryNode(object):
    """HGF binary state node"""
    def __init__(self):

        # Initialize parent
        self.pa = None

        # Initialize time series
        self.reset()

    def reset(self):
        self.times = [0.0]
        self.pihats = [None]
        self.pis = [None]
        self.muhats = [None]
        self.mus = [None]

    def set_parent(self, *, parent):
        self.pa = parent

    def new_muhat_pihat(self, time):
        muhat_pa = self.pa.new_muhat(time)
        muhat = sgm(muhat_pa)
        pihat = 1 / (muhat * (1 - muhat))
        return [muhat, pihat]

    def vape(self):
        return self.mus[-1] - self.muhats[-1]

    def update_parent(self, time):
        pa = self.pa

        if not pa:
            return

        pihat = self.pihats[-1]

        # Update parent
        vape = self.vape()

        pihat_pa, nu_pa = pa.new_pihat_nu(time)
        pi_pa = pihat_pa + 1 / pihat

        muhat_pa = pa.new_muhat(time)
        mu_pa = muhat_pa + vape / pi_pa

        pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

    def update(self, time, pihat, pi, muhat, mu):
        self.times.append(time)
        self.pihats.append(pihat)
        self.pis.append(pi),
        self.muhats.append(muhat)
        self.mus.append(mu)

        self.update_parent(time)


# HGF continuous input nodes
class InputNode(object):
    """An HGF node that receives input on a continuous scale"""
    def __init__(self, *, omega):

        # Incorporate parameter attributes
        self.omega = omega
        self.kappa = None

        # Initialize parents
        self.va_pa = None
        self.vo_pa = None

        # Initialize time series
        self.reset()

    def reset(self):
        self.times = [0.0]
        self.inputs = [None]
        self.surprises = [0.0]

    def set_value_parent(self, *, parent):
        self.va_pa = parent

    def set_volatility_parent(self, *, parent, kappa):
        self.vo_pa = parent
        self.kappa = kappa

    # Update parents and return surprise
    def update_parents(self, value, time):
        va_pa = self.va_pa
        vo_pa = self.vo_pa

        lognoise = self.omega
        kappa = self.kappa

        if kappa is not None:
            lognoise += kappa * vo_pa.mu[-1]

        pihat = 1 / np.exp(lognoise)

        # Update value parent
        pihat_va_pa, nu_va_pa = va_pa.new_pihat_nu(time)
        pi_va_pa = pihat_va_pa + pihat

        muhat_va_pa = va_pa.new_muhat(time)
        vape = value - muhat_va_pa
        mu_va_pa = muhat_va_pa + pihat / pi_va_pa * vape

        va_pa.update(time, pihat_va_pa, pi_va_pa, muhat_va_pa,
                     mu_va_pa, nu_va_pa)

        # Update volatility parent
        if vo_pa is not None:
            vope = (1 / pi_va_pa + (value - mu_va_pa)**2) * pihat - 1

            pihat_vo_pa, nu_vo_pa = vo_pa.new_pihat_nu(time)
            pi_vo_pa = pihat_vo_pa + 0.5 * kappa**2 * (1 + vope)
            if pi_vo_pa <= 0:
                raise NegativePosteriorPrecisionError(
                    'Parameters values are in region where model ' +
                    'assumptions are violated.')

            muhat_vo_pa = vo_pa.new_muhat(time)
            mu_vo_pa = muhat_vo_pa + 0.5 * kappa / pi_vo_pa * vope

            vo_pa.update(time, pihat_vo_pa, pi_vo_pa, muhat_vo_pa,
                         mu_vo_pa, nu_vo_pa)

        return gaussian_surprise(value, muhat_va_pa, pihat)

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.surprises.append(self.update_parents(value, time))

    def input(self, inputs):
        try:
            for input in inputs:
                try:
                    value = input[0]
                    time = input[1]
                except IndexError:
                    value = input
                    time = self.times[-1] + 1
                    self._single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self._single_input(value, time)


# HGF binary input nodes
class BinaryInputNode(object):
    """An HGF node that receives binary input"""
    def __init__(self,
                 *,
                 pihat=np.inf,
                 eta0=0.0,
                 eta1=1.0):

        # Incorporate parameter attributes
        self.pihat = pihat
        self.eta0 = eta0
        self.eta1 = eta1

        # Initialize parent
        self.pa = None

        # Initialize time series
        self.reset()

    def reset(self):
        self.times = [0.0]
        self.inputs = [None]
        self.surprises = [0.0]

    def set_parent(self, *, parent):
        self.pa = parent

    def update_parent(self, value, time):
        pa = self.pa
        surprise = 0.0

        pihat = self.pihat

        muhat_pa, pihat_pa = pa.new_muhat_pihat(time)

        if pihat == np.inf:
            # Just pass the value through in the absence of noise
            mu_pa = value
            pi_pa = np.inf
            surprise = binary_surprise(value, muhat_pa)
        else:
            # Likelihood under eta1
            und1 = np.exp(-pihat / 2 * (value - self.eta1)**2)
            # Likelihood under eta0
            und0 = np.exp(-pihat / 2 * (value - self.eta0)**2)
            # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
            mu_pa = muhat_pa * und1 / (muhat_pa * und1 + (1 - muhat_pa) * und0)
            pi_pa = 1 / (mu_pa * (1 - mupa))
            # Surprise
            surprise = (-np.log(muhat_pa * gaussian(value, self.eta1, pihat) +
                        (1 - muhat_pa) * gaussian(value, self.eta0, pihat)))

        pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa)

        return surprise

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.surprises.append(self.update_parent(value, time))

    def input(self, inputs):
        try:
            for input in inputs:
                try:
                    value = input[0]
                    time = input[1]
                except IndexError:
                    value = input
                    time = self.times[-1] + 1
                    self._single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self._single_input(value, time)


# Standard 2-level HGF for continuous inputs
class StandardHGF(object):
    """The standard 2-level HGF for continuous inputs"""
    def __init__(self,
                 *,
                 initial_mu1,
                 initial_pi1,
                 initial_mu2,
                 initial_pi2,
                 omega1,
                 kappa1,
                 omega2,
                 omega_input,
                 rho1=0.0,
                 rho2=0.0,
                 phi1=0.0,
                 m1=0.0,
                 phi2=0.0,
                 m2=0.0):

        # Set up nodes and their relationships
        self.x2 = StateNode(initial_mu=initial_mu2,
                            initial_pi=initial_pi2,
                            omega=omega2,
                            rho=rho2,
                            phi=phi2,
                            m=m2)
        self.x1 = StateNode(initial_mu=initial_mu1,
                            initial_pi=initial_pi1,
                            omega=omega1,
                            rho=rho1,
                            phi=phi1,
                            m=m1)
        self.xU = InputNode(omega=omega_input)

        self.x1.add_volatility_parent(parent=self.x2, kappa=kappa1)
        self.xU.set_value_parent(parent=self.x1)

    def reset(self):
        self.x2.reset()
        self.x1.reset()
        self.xU.reset()

    def input(self, inputs):
        self.xU.input(inputs)

    def surprise(self, inputs):
        return sum(self.xU.surprises)


# Standard 3-level HGF for binary inputs
class StandardBinaryHGF(object):
    """The standard 3-level HGF for binary inputs"""
    def __init__(self,
                 *,
                 initial_mu2,
                 initial_pi2,
                 initial_mu3,
                 initial_pi3,
                 omega2,
                 kappa2,
                 omega3,
                 pihat_input=np.inf,
                 eta0=0.0,
                 eta1=1.0,
                 rho2=0.0,
                 rho3=0.0,
                 phi2=0.0,
                 m2=0.0,
                 phi3=0.0,
                 m3=0.0):

        # Set up nodes and their relationships
        self.x3 = StateNode(initial_mu=initial_mu3,
                            initial_pi=initial_pi3,
                            omega=omega3,
                            rho=rho3,
                            phi=phi3,
                            m=m3)
        self.x2 = StateNode(initial_mu=initial_mu2,
                            initial_pi=initial_pi2,
                            omega=omega2,
                            rho=rho2,
                            phi=phi2,
                            m=m2)
        self.x1 = BinaryNode()
        self.xU = BinaryInputNode(pihat=pihat_input,
                                  eta0=eta0,
                                  eta1=eta1)

        self.x2.add_volatility_parent(parent=self.x3, kappa=kappa2)
        self.x1.set_parent(parent=self.x2)
        self.xU.set_parent(parent=self.x1)

    def reset(self):
        self.x3.reset()
        self.x2.reset()
        self.x1.reset()
        self.xU.reset()

    def input(self, inputs):
        self.xU.input(inputs)

    def surprise(self, inputs):
        return sum(self.xU.surprises)


class Parameter(object):
    def __init__(self,
                 *,
                 space='native',
                 lower_bound=None,
                 upper_bound=None,
                 value=None,
                 trans_value=None,
                 prior_mean=None,
                 trans_prior_mean=None,
                 trans_prior_precision=None):

        # Initialize attributes
        self.space = space
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if value is not None and trans_value is not None:
            raise ParameterConfigurationError(
                'Only one of value and trans_value can be given.')
        elif value is not None:
            self.value = value
        elif trans_value is not None:
            self.trans_value = trans_value
        else:
            raise ParameterConfigurationError(
                'One of value and trans_value must be given.')

        if prior_mean is not None and trans_prior_mean is not None:
            raise ParameterConfigurationError(
                'Only one of prior_mean and trans_prior_mean can be given.')
        elif prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.trans_prior_mean = trans_prior_mean

        if (trans_prior_precision is None and
            (prior_mean is not None or
             trans_prior_mean is not None)):
            raise ParameterConfigurationError(
                'trans_prior_precision must be given if prior_mean ' +
                'or trans_prior_mean is given')
        else:
            self.trans_prior_precision = trans_prior_precision

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space):
        if space is 'native':
            self._space = space
            self._lower_bound = None
            self._upper_bound = None
        elif space is 'log':
            self._space = space
            self._lower_bound = 0
            self._upper_bound = None
        elif space is 'logit':
            self._space = space
            self._lower_bound = 0
            self._upper_bound = 1
        else:
            raise ParameterConfigurationError(
                "Space must be one of 'native, 'log', or 'logit'")

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        space = self.space
        if lower_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "lower_bound must be None if space is 'native'.")
        elif space is 'log':
            self._lower_bound = lower_bound
            self._upper_bound = None
        else:
            self._lower_bound = lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        space = self.space
        if upper_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "upper_bound must be None if space is 'native'.")
        elif space is 'log':
            self._lower_bound = None
            self._upper_bound = upper_bound
        else:
            self._upper_bound = upper_bound

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

        space = self.space
        if space is 'native':
            self._trans_value = value
        elif space is 'log':
            self._trans_value = log(value,
                                    lower_bound=self.lower_bound,
                                    upper_bound=self.upper_bound)
        elif space is 'logit':
            self._trans_value = logit(value,
                                      lower_bound=self.lower_bound,
                                      upper_bound=self.upper_bound)

    @property
    def trans_value(self):
        return self._trans_value

    @trans_value.setter
    def trans_value(self, trans_value):
        self._trans_value = trans_value

        space = self.space
        if space is 'native':
            self._value = trans_value
        elif space is 'log':
            self._value = exp(trans_value,
                              lower_bound=self.lower_bound,
                              upper_bound=self.upper_bound)
        elif space is 'logit':
            self._value = sgm(trans_value,
                              lower_bound=self.lower_bound,
                              upper_bound=self.upper_bound)

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean):
        self._prior_mean = prior_mean

        if prior_mean is not None:
            space = self.space
            if space is 'native':
                self._trans_prior_mean = prior_mean
            elif space is 'log':
                self._trans_prior_mean = log(prior_mean,
                                             lower_bound=self.lower_bound,
                                             upper_bound=self.upper_bound)
            elif space is 'logit':
                self._trans_prior_mean = logit(prior_mean,
                                               lower_bound=self.lower_bound,
                                               upper_bound=self.upper_bound)
        else:
            self._trans_prior_mean = None
            self._trans_prior_precision = None

    @property
    def trans_prior_mean(self):
        return self._trans_prior_mean

    @trans_prior_mean.setter
    def trans_prior_mean(self, trans_prior_mean):
        self._trans_prior_mean = trans_prior_mean

        if trans_prior_mean is not None:
            space = self.space
            if space is 'native':
                self._prior_mean = trans_prior_mean
            elif space is 'log':
                self._prior_mean = exp(trans_prior_mean,
                                       lower_bound=self.lower_bound,
                                       upper_bound=self.upper_bound)
            elif space is 'logit':
                self._prior_mean = sgm(trans_prior_mean,
                                       lower_bound=self.lower_bound,
                                       upper_bound=self.upper_bound)
        else:
            self._prior_mean = None
            self._trans_prior_precision = None

    @property
    def trans_prior_precision(self):
        return self._trans_prior_precision

    @trans_prior_precision.setter
    def trans_prior_precision(self, trans_prior_precision):
        self._trans_prior_precision = trans_prior_precision

        if trans_prior_precision is None:
            self._prior_mean = None
            self._trans_prior_mean = None


def exp(x, *, lower_bound=0.0, upper_bound=None):
    """The (shifted and mirrored) exponential function"""
    if upper_bound is not None:
        return -np.exp(x) + upper_bound
    else:
        return np.exp(x) + lower_bound


def log(x, *, lower_bound=0.0, upper_bound=None):
    """The (shifted and mirrored) natural logarithm"""
    if upper_bound is not None:
        return np.log(-x + upper_bound)
    else:
        return np.log(x - lower_bound)


def sgm(x, *, lower_bound=0.0, upper_bound=1.0):
    """The logistic sigmoid function"""
    return (upper_bound - lower_bound) / (1 + np.exp(-x)) + lower_bound


def logit(x, *, lower_bound=0.0, upper_bound=1.0):
    """The logistic function"""
    if x < lower_bound:
        raise LogitArgumentError('Logit argmument may not be less than ' +
                                 'lower bound')
    if x > upper_bound:
        raise LogitArgumentError('Logit argmument may not be greater than ' +
                                 'upper bound')

    if x == lower_bound:
        return -np.inf
    elif x == upper_bound:
        return np.inf
    else:
        return np.log((x - lower_bound) / (upper_bound - x))


def gaussian(x, mu, pi):
    """The Gaussian density as defined by mean and precision"""
    return pi / np.sqrt(2 * np.pi) * np.exp(-pi / 2 * (x - mu)**2)


def gaussian_surprise(x, muhat, pihat):
    """Surprise at an outcome under a Gaussian prediction"""
    return 0.5 * (np.log(2 * np.pi) - np.log(pihat) + pihat * (x - muhat)**2)


def binary_surprise(x, muhat):
    """Surprise at a binary outcome"""
    if x == 1:
        return -np.log(1 - muhat)
    if x == 0:
        return -np.log(muhat)
    else:
        raise OutcomeValueError('Outcome needs to be either 0 or 1.')


class HgfException(Exception):
    """Base class for all exceptions raised by the hgf module."""


class NodeConfigurationError(HgfException):
    """Node configuration error."""


class ParameterConfigurationError(HgfException):
    """Parameter configuration error."""


class NegativePosteriorPrecisionError(HgfException):
    """Negative posterior precision."""


class OutcomeValueError(HgfException):
    """Outcome value error."""


class LogitArgumentError(HgfException):
    """Logit argument out of bounds."""
