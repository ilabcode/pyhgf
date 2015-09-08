# hgf.py
"""The HGF time series model."""

import numpy as np
import warnings


# Turn warnings into exceptions
warnings.simplefilter('error')


# HGF state nodes
class StateNode(object):
    """The basic unit of an HGF model."""
    def __init__(self,
                 *,
                 prior_mu,
                 prior_pi,
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
        self.prior_mu = prior_mu
        self.prior_pi = prior_pi
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
        self.times = [0.0]
        self.pihats = [None]
        self.pis = [self.prior_pi]
        self.muhats = [None]
        self.mus = [self.prior_mu]
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


# HGF input nodes
class InputNode(object):
    """A node that receives input on a continuous scale."""
    def __init__(self, *, omega):

        # Incorporate parameter attributes
        self.omega = omega
        self.kappa = None

        # Initialize parents
        self.va_pa = None
        self.vo_pa = None

        # Initialize time series
        self.times = [0.0]
        self.inputs = [None]

    def set_value_parent(self, *, parent):
        self.va_pa = parent

    def set_volatility_parent(self, *, parent, kappa):
        self.vo_pa = parent
        self.kappa = kappa

    def update_parents(self, input, time):
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
        vape = input - muhat_va_pa
        mu_va_pa = muhat_va_pa + pihat / pi_va_pa * vape

        va_pa.update(time, pihat_va_pa, pi_va_pa, muhat_va_pa,
                     mu_va_pa, nu_va_pa)

        # Update volatility parent
        if vo_pa is not None:
            vope = (1 / pi_va_pa + (input - mu_va_pa)**2) * pihat - 1

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

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.update_parents(value, time)

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
                 prior_mu1,
                 prior_pi1,
                 prior_mu2,
                 prior_pi2,
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

        # Initialize parameter attributes
        self.prior_mu1 = prior_mu1
        self.prior_pi1 = prior_pi1
        self.prior_mu2 = prior_mu2
        self.prior_pi2 = prior_pi2
        self.omega1 = omega1
        self.kappa1 = kappa1
        self.omega2 = omega2
        self.omega_input = omega_input
        self.rho1 = rho1
        self.rho2 = rho2
        self.phi1 = phi1
        self.m1 = m1
        self.phi2 = phi2
        self.m2 = m2

        # Set up nodes and their relationships
        self.x2 = StateNode(prior_mu=prior_mu2,
                            prior_pi=prior_pi2,
                            omega=omega2,
                            rho=rho2,
                            phi=phi2,
                            m=m2)
        self.x1 = StateNode(prior_mu=prior_mu1,
                            prior_pi=prior_pi1,
                            omega=omega1,
                            rho=rho1,
                            phi=phi1,
                            m=m1)
        self.xU = InputNode(omega=omega_input)

        self.x1.add_volatility_parent(parent=self.x2, kappa=kappa1)
        self.xU.set_value_parent(parent=self.x1)

    def input(self, inputs):
        self.xU.input(inputs)


class HgfException(Exception):
    """Base class for all exceptions raised by the hgf module."""


class NodeConfigurationError(HgfException):
    """Node configuration error."""


class NegativePosteriorPrecisionError(HgfException):
    """Negative posterior precision."""
