import numpy as np


# Parameters for HGF state nodes
class StateNodeParameters(object):
    def __init__(self,
                 priorMean=None,
                 priorPrecision=None,
                 rho=0.0,
                 phis=[],
                 omega=0.0,
                 kappas=[]):

        # Collect arguments
        self.priorMean = priorMean
        self.priorPrecision = priorPrecision
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
                 parameters,
                 valueParents=[],
                 volatilityParents=[]):

        # Sanity checks
        if len(parameters.phis) != len(valueParents):
            raise ValueError('hgf.StateNode: lengths of phis and ' +
                             'valueParents must match.')

        if len(parameters.kappas) != len(volatilityParents):
            raise ValueError('hgf.StateNode: lengths of kappas and ' +
                             'volatilityParents must match.')

        # Collect parents
        self.vApas = valueParents
        self.vOpas = volatilityParents

        # Incorporate parameters
        self.priorMean = parameters.priorMean
        self.priorPrecision = parameters.priorPrecision
        self.rho = parameters.rho
        self.phis = parameters.phis
        self.omega = parameters.omega
        self.kappas = parameters.kappas

        # Initialize time series
        self.times = [0.0]
        self.pihats = [None]
        self.pis = [self.priorPrecision]
        self.muhats = [None]
        self.mus = [self.priorMean]
        self.nus = [None]

    def newMuhat(self, time):
        t = time - self.times[-1]
        driftrate = self.rho
        for i in range(len(self.vApas)):
            driftrate += self.phis[i] * self.vApas[i].mus[-1]
        return self.mus[-1] + t * driftrate

    def newNu(self, time):
        t = time - self.times[-1]
        logvol = self.omega
        for i in range(len(self.vOpas)):
            logvol += self.kappas[i] * self.vOpas[i].mus[-1]
        return t * np.exp(logvol)

    def newPihatNu(self, time):
        newNu = self.newNu(time)
        return [1 / (1 / self.pis[-1] + newNu), newNu]

    def vApe(self):
        return self.mus[-1] - self.muhats[-1]

    def vOpe(self):
        return ((1 / self.pis[-1] + self.vApe()**2) *
                self.pihats[-1] - 1)

    def updateParents(self, time):
        vApas = self.vApas
        vOpas = self.vOpas

        if len(vApas + vOpas) == 0:
            return

        pihat = self.pihats[-1]

        # Update value parents
        phis = self.phis
        vApe = self.vApe()

        for i in range(len(vApas)):
            pihatPa, nuPa = vApas[i].newPihatNu(time)
            piPa = pihatPa + phis[i]**2 * pihat

            muhatPa = vApas[i].newMuhat
            muPa = muhatPa + phis[i] * pihat / piPa * vApe

            vApas[i].update(time, pihatPa, piPa, muhatPa, muPa, nuPa)

        # Update volatility parents
        nu = self.nus[-1]
        kappas = self.kappas
        vOpe = self.vOpe()

        for i in range(len(vOpas)):
            pihatPa, nuPa = vOpas[i].newPihatNu(time)
            piPa = pihatPa + 0.5 * (kappas[i] * nu * pihat)**2 * \
                (1 + (1 - 1 / (nu * self.pis[-2])) * vOpe)

            muhatPa = vOpas[i].newMuhat(time)
            muPa = muhatPa + 0.5 * kappas[i] * nu * pihat / piPa * vOpe

            vOpas[i].update(time, pihatPa, piPa, muhatPa, muPa, nuPa)

    def update(self, time, pihat, pi, muhat, mu, nu):
        self.times.append(time)
        self.pihats.append(pihat)
        self.pis.append(pi),
        self.muhats.append(muhat)
        self.mus.append(mu)
        self.nus.append(nu)

        self.updateParents(time)


# HGF input nodes
class InputNode(object):
    def __init__(self,
                 parameters,
                 valueParent,
                 volatilityParent=None):

        # Sanity check
        if ((parameters.kappa is None and volatilityParent is not None) or
           (volatilityParent is None and parameters.kappa is not None)):
            raise ValueError('hgf.InputNode: kappa and volatilityParent ' +
                             'must either be both None or both defined.')

        # Collect parents
        self.vApa = valueParent
        self.vOpa = volatilityParent

        # Incorporate parameters
        self.omega = parameters.omega
        self.kappa = parameters.kappa

        # Initialize time series
        self.times = [0.0]
        self.inputs = [None]

    def updateParents(self, input, time):
        vApa = self.vApa
        vOpa = self.vOpa

        lognoise = self.omega
        kappa = self.kappa

        if kappa is not None:
            lognoise += kappa * vOpa.mu[-1]

        pihat = 1 / np.exp(lognoise)

        # Update value parent
        pihatVApa, nuVApa = vApa.newPihatNu(time)
        piVApa = pihatVApa + pihat

        muhatVApa = vApa.newMuhat(time)
        vApe = input - muhatVApa
        muVApa = muhatVApa + pihat / piVApa * vApe

        vApa.update(time, pihatVApa, piVApa, muhatVApa, muVApa, nuVApa)

        # Update volatility parent
        if vOpa is not None:
            vOpe = (1 / piVApa + (input - muVApa)**2) * pihat - 1

            pihatVOpa, nuVOpa = vOpa.newPihatNu(time)
            piVOpa = pihatVOpa + 0.5 * kappa**2 * (1 + vOpe)

            muhatVOpa = vOpa.newMuhat(time)
            muVOpa = muhatVOpa + 0.5 * kappa / piVOpa * vOpe

            vOpa.update(time, pihatVOpa, piVOpa, muhatVOpa, muVOpa, nuVOpa)

    def input(self, input, time=-1):
        if time == -1:
            time = self.times[-1] + 1

        self.times.append(time)
        self.inputs.append(input)

        self.updateParents(input, time)
