import numpy as np


# Parameters for HGF state nodes
class StateNodeParameters(object):
    def __init__(self,
                 prior_mu=None,
                 prior_pi=None,
                 rho=0.0,
                 phis=[],
                 omega=0.0,
                 kappas=[]):

        # Collect arguments
        self.prior_mu = prior_mu
        self.prior_pi = prior_pi
        self.omega = omega
        self.kappas = kappas
        self.rho = rho
        self.phis = phis


# Parameters for HGF state nodes
class InputNodeParameters(object):
    def __init__(self,
                 omega=0.0,
                 kappa=None):

        # Collect arguments
        self.omega = omega
        self.kappa = kappa


# HGF state nodes
class StateNode(object):
    def __init__(self,
                 state_node_params,
                 value_parents=[],
                 volatility_parents=[]):

        # Sanity checks
        if len(state_node_params.phis) != len(value_parents):
            raise ValueError('hgf.StateNode: lengths of phis and ' +
                             'value_parents must match.')

        if len(state_node_params.kappas) != len(volatility_parents):
            raise ValueError('hgf.StateNode: lengths of kappas and ' +
                             'volatility_parents must match.')

        # Collect parents
        self.va_pas = value_parents
        self.vo_pas = volatility_parents

        # Incorporate state_node_params
        self.prior_mu = state_node_params.prior_mu
        self.prior_pi = state_node_params.prior_pi
        self.rho = state_node_params.rho
        self.phis = state_node_params.phis
        self.omega = state_node_params.omega
        self.kappas = state_node_params.kappas

        # Initialize time series
        self.times = [0.0]
        self.pihats = [None]
        self.pis = [self.prior_pi]
        self.muhats = [None]
        self.mus = [self.prior_mu]
        self.nus = [None]

    def new_muhat(self, time):
        t = time - self.times[-1]
        driftrate = self.rho
        for i, va_pa in enumerate(self.va_pas):
            driftrate += self.phis[i] * self.va_pas[i].mus[-1]
        return self.mus[-1] + t * driftrate

    def new_nu(self, time):
        t = time - self.times[-1]
        logvol = self.omega
        for i, vo_pa in enumerate(self.vo_pas):
            logvol += self.kappas[i] * self.vo_pas[i].mus[-1]
        return t * np.exp(logvol)

    def new_pihat_nu(self, time):
        new_nu = self.new_nu(time)
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
        phis = self.phis
        vape = self.vape()

        for i, va_pa in enumerate(va_pas):
            pihat_pa, nu_pa = va_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + phis[i]**2 * pihat

            muhat_pa = va_pa.new_muhat
            mu_pa = muhat_pa + phis[i] * pihat / pi_pa * vape

            va_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

        # Update volatility parents
        nu = self.nus[-1]
        kappas = self.kappas
        vope = self.vope()

        for i, vo_pa in enumerate(vo_pas):
            pihat_pa, nu_pa = vo_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + 0.5 * (kappas[i] * nu * pihat)**2 * \
                (1 + (1 - 1 / (nu * self.pis[-2])) * vope)

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
    def __init__(self,
                 parameters,
                 value_parent,
                 volatility_parent=None):

        # Sanity check
        if ((parameters.kappa is None and volatility_parent is not None) or
           (volatility_parent is None and parameters.kappa is not None)):
            raise ValueError('hgf.InputNode: kappa and volatility_parent ' +
                             'must either be both None or both defined.')

        # Collect parents
        self.va_pa = value_parent
        self.vo_pa = volatility_parent

        # Incorporate parameters
        self.omega = parameters.omega
        self.kappa = parameters.kappa

        # Initialize time series
        self.times = [0.0]
        self.inputs = [None]

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

            muhat_vo_pa = vo_pa.new_muhat(time)
            mu_vo_pa = muhat_vo_pa + 0.5 * kappa / pi_vo_pa * vope

            vo_pa.update(time, pihat_vo_pa, pi_vo_pa, muhat_vo_pa,
                         mu_vo_pa, nu_vo_pa)

    def input(self, input, time=-1):
        if time == -1:
            time = self.times[-1] + 1

        self.times.append(time)
        self.inputs.append(input)

        self.update_parents(input, time)
