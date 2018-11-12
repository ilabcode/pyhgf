# HGF continuous state node
class StateNode(object):
    """HGF continuous state node"""
    def __init__(self,
                 *,
                 initial_mu,
                 initial_pi,
                 rho=0,
                 phi=0,
                 m=0,
                 omega=0):

        # Sanity check
        if rho and phi:
            raise NodeConfigurationError(
                'hgf.StateNode: rho (drift) and phi (AR(1) parameter) may ' +
                'not be non-zero at the same time.')

        # Initialize parameter attributes
        self.initial_mu = Parameter(value=initial_mu)
        self.initial_pi = Parameter(value=initial_pi, space='log')
        self.rho = Parameter(value=rho)
        self.phi = Parameter(value=phi, space='logit')
        self.m = Parameter(value=m)
        self.omega = Parameter(value=omega)
        self.psis = []
        self.kappas = []

        # Initialize parents
        self.va_pas = []
        self.vo_pas = []

        # Initialize time series
        self.times = [0]
        self.pihats = [None]
        self.pis = [self.initial_pi.value]
        self.muhats = [None]
        self.mus = [self.initial_mu.value]
        self.nus = [None]

    @property
    def parents(self):
        parents = []
        parents.extend(self.va_pas)
        parents.extend(self.vo_pas)
        return parents

    @property
    def params(self):
        params = [self.initial_mu,
                  self.initial_pi,
                  self.rho,
                  self.phi,
                  self.m,
                  self.omega]

        params.extend(self.psis)
        params.extend(self.kappas)

        return params

    def reset(self):
        self._times_backup = self.times
        self.times = [0]

        self._pihats_backup = self.pihats
        self.pihats = [None]

        self._pis_backup = self.pis
        self.pis = [self.initial_pi.value]

        self._muhats_backup = self.muhats
        self.muhats = [None]

        self._mus_backup = self.mus
        self.mus = [self.initial_mu.value]

        self._nus_backup = self.nus
        self.nus = [None]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.pihats = self._pihats_backup
        self.pis = self._pis_backup
        self.muhats = self._muhats_backup
        self.mus = self._mus_backup
        self.nus = self._nus_backup

    def reset_hierarchy(self):
        self.reset()
        for pa in self.parents:
            pa.reset_hierarchy()

    def undo_last_reset_hierarchy(self):
        self.undo_last_reset()
        for pa in self.parents:
            pa.undo_last_reset_hierarchy()

    def add_value_parent(self, *, parent, psi):
        self.va_pas.append(parent)
        self.psis.append(Parameter(value=psi))

    def add_volatility_parent(self, *, parent, kappa):
        self.vo_pas.append(parent)
        self.kappas.append(Parameter(value=kappa, space='log'))

    def new_muhat(self, time):
        t = time - self.times[-1]
        driftrate = self.rho.value
        for i, va_pa in enumerate(self.va_pas):
            driftrate += self.psis[i].value * self.va_pas[i].mus[-1]
        return self.mus[-1] + t * driftrate

    def _new_nu(self, time):
        t = time - self.times[-1]
        logvol = self.omega.value
        for i, vo_pa in enumerate(self.vo_pas):
            logvol += self.kappas[i].value * self.vo_pas[i].mus[-1]
        nu = t * np.exp(logvol)
        if nu > 1e-128:
            return nu
        else:
            raise HgfUpdateError(
                'Nu is zero. Parameters values are in region where model\n' +
                'assumptions are violated.')

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
            pi_pa = pihat_pa + psis[i].value**2 * pihat

            muhat_pa = va_pa.new_muhat(time)
            mu_pa = muhat_pa + psis[i].value * pihat / pi_pa * vape

            va_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

        # Update volatility parents
        nu = self.nus[-1]
        kappas = self.kappas
        vope = self.vope()

        for i, vo_pa in enumerate(vo_pas):
            pihat_pa, nu_pa = vo_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + 0.5 * (kappas[i].value * nu * pihat)**2 * \
                (1 + (1 - 1 / (nu * self.pis[-2])) * vope)
            if pi_pa <= 0:
                raise HgfUpdateError(
                    'Negative posterior precision. Parameters values are\n' +
                    'in a region where model assumptions are violated.')

            muhat_pa = vo_pa.new_muhat(time)
            mu_pa = (muhat_pa +
                     0.5 * kappas[i].value * nu * pihat / pi_pa * vope)

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
        self.times = [0]
        self.pihats = [None]
        self.pis = [None]
        self.muhats = [None]
        self.mus = [None]

    @property
    def parents(self):
        parents = []
        if self.pa:
            parents.append(self.pa)
        return parents

    @property
    def params(self):
        return []

    def reset(self):
        self._times_backup = self.times
        self.times = [0]

        self._pihats_backup = self.pihats
        self.pihats = [None]

        self._pis_backup = self.pis
        self.pis = [None]

        self._muhats_backup = self.muhats
        self.muhats = [None]

        self._mus_backup = self.mus
        self.mus = [None]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.pihats = self._pihats_backup
        self.pis = self._pis_backup
        self.muhats = self._muhats_backup
        self.mus = self._mus_backup

    def reset_hierarchy(self):
        self.reset()
        for pa in self.parents:
            pa.reset_hierarchy()

    def undo_last_reset_hierarchy(self):
        self.undo_last_reset()
        for pa in self.parents:
            pa.undo_last_reset_hierarchy()

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
        self.omega = Parameter(value=omega)
        self.kappa = None

        # Initialize parents
        self.va_pa = None
        self.vo_pa = None

        # Initialize time series
        self.times = [0]
        self.inputs = [None]
        self.inputs_with_times = [(None, 0)]
        self.surprises = [0]

    @property
    def parents(self):
        parents = []
        if self.va_pa:
            parents.append(self.va_pa)
        if self.vo_pa:
            parents.append(self.vo_pa)
        return parents

    @property
    def params(self):
        params = [self.omega]

        if self.kappa is not None:
            params.append(self.kappa)

        return params

    def reset(self):
        self._times_backup = self.times
        self.times = [0]

        self._inputs_backup = self.inputs
        self.inputs = [None]

        self._inputs_with_times_backup = self.inputs_with_times
        self.inputs_with_times = [(None, 0)]

        self._surprises_backup = self.surprises
        self.surprises = [0]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.inputs = self._inputs_backup
        self.inputs_with_times = self._inputs_with_times_backup
        self.surprises = self._surprises_backup

    def reset_hierarchy(self):
        self.reset()
        for pa in self.parents:
            pa.reset_hierarchy()

    def undo_last_reset_hierarchy(self):
        self.undo_last_reset()
        for pa in self.parents:
            pa.undo_last_reset_hierarchy()

    def recalculate(self):
        iwt = list(self.inputs_with_times[1:])
        self.reset_hierarchy()
        try:
            self.input(iwt)
        except HgfUpdateError as e:
            self.undo_last_reset_hierarchy()
            raise e

    def set_value_parent(self, *, parent):
        self.va_pa = parent

    def set_volatility_parent(self, *, parent, kappa):
        self.vo_pa = parent
        self.kappa = Parameter(value=kappa, space='log')

    # Update parents and return surprise
    def update_parents(self, value, time):
        va_pa = self.va_pa
        vo_pa = self.vo_pa

        if not vo_pa and not va_pa:
            return

        lognoise = self.omega.value

        kappa = None
        if self.kappa is not None:
            kappa = self.kappa.value

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
                raise HgfUpdateError(
                    'Negative posterior precision. Parameters values are\n' +
                    'in a region where model assumptions are violated.')

            muhat_vo_pa = vo_pa.new_muhat(time)
            mu_vo_pa = muhat_vo_pa + 0.5 * kappa / pi_vo_pa * vope

            vo_pa.update(time, pihat_vo_pa, pi_vo_pa, muhat_vo_pa,
                         mu_vo_pa, nu_vo_pa)

        return gaussian_surprise(value, muhat_va_pa, pihat)

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.inputs_with_times.append((value, time))
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
                 eta0=0,
                 eta1=1):

        # Incorporate parameter attributes
        self.pihat = Parameter(value=pihat, space='log')
        self.eta0 = Parameter(value=eta0)
        self.eta1 = Parameter(value=eta1)

        # Initialize parent
        self.pa = None

        # Initialize time series
        self.times = [0]
        self.inputs = [None]
        self.inputs_with_times = [(None, 0)]
        self.surprises = [0]

    @property
    def parents(self):
        parents = []
        if self.pa is not None:
            parents.append(self.pa)
        return parents

    @property
    def params(self):
        return [self.pihat,
                self.eta0,
                self.eta1]

    def reset(self):
        self._times_backup = self.times
        self.times = [0]

        self._inputs_backup = self.inputs
        self.inputs = [None]

        self._inputs_with_times_backup = self.inputs_with_times
        self.inputs_with_times = [(None, 0)]

        self._surprises_backup = self.surprises
        self.surprises = [0]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.inputs = self._inputs_backup
        self.inputs_with_times = self._inputs_with_times_backup
        self.surprises = self._surprises_backup

    def reset_hierarchy(self):
        self.reset()
        for pa in self.parents:
            pa.reset_hierarchy()

    def undo_last_reset_hierarchy(self):
        self.undo_last_reset()
        for pa in self.parents:
            pa.undo_last_reset_hierarchy()

    def recalculate(self):
        iwt = list(self.inputs_with_times[1:])
        self.reset_hierarchy()
        try:
            self.input(iwt)
        except HgfUpdateError as e:
            self.undo_last_reset_hierarchy()
            raise e

    def set_parent(self, *, parent):
        self.pa = parent

    def update_parent(self, value, time):
        pa = self.pa
        if not pa:
            return

        surprise = 0

        pihat = self.pihat.value

        muhat_pa, pihat_pa = pa.new_muhat_pihat(time)

        if pihat == np.inf:
            # Just pass the value through in the absence of noise
            mu_pa = value
            pi_pa = np.inf
            surprise = binary_surprise(value, muhat_pa)
        else:
            eta1 = self.eta1.value
            eta0 = self.eta0.value
            # Likelihood under eta1
            und1 = np.exp(-pihat / 2 * (value - eta1)**2)
            # Likelihood under eta0
            und0 = np.exp(-pihat / 2 * (value - eta0)**2)
            # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
            mu_pa = muhat_pa * und1 / (muhat_pa * und1 + (1 - muhat_pa) * und0)
            pi_pa = 1 / (mu_pa * (1 - mu_pa))
            # Surprise
            surprise = (-np.log(muhat_pa * gaussian(value, eta1, pihat) +
                        (1 - muhat_pa) * gaussian(value, eta0, pihat)))

        pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa)

        return surprise

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.inputs_with_times.append((value, time))
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
                finally:
                    self._single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self._single_input(value, time)

