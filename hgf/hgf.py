# hgf.py
"""The HGF time series model."""

import numpy as np
import warnings


# Turn warnings into exceptions
warnings.simplefilter('error')


# Generic HGF model
class Model(object):
    """Generic HGF model"""
    def __init__(self):
        self._nodes = []

    @property
    def nodes(self):
        return self._nodes

    @property
    def input_nodes(self):
        input_nodes = []
        for node in self.nodes:
            if (isinstance(node, InputNode) or
                    isinstance(node, BinaryInputNode)):
                input_nodes.append(node)
        return input_nodes

    @property
    def params(self):
        params = []
        for node in self.nodes:
            params.extend(node.params)
        return params

    @property
    def var_params(self):
        var_params = []
        for param in self.params:
            tpp = param.trans_prior_precision
            if tpp is not None and tpp is not np.inf:
                var_params.append(param)
        return var_params

    @property
    def param_values(self):
        return [param.value for param in self.params]

    @param_values.setter
    def param_values(self, values):
        for i, param in enumerate(self.params):
            param.value = values[i]

    @property
    def var_param_values(self):
        return [var_param.value for var_param in self.var_params]

    @var_param_values.setter
    def var_param_values(self, values):
        for i, var_param in enumerate(self.var_params):
            var_param.value = values[i]

    @property
    def param_trans_values(self):
        return [param.trans_value for param in self.params]

    @param_trans_values.setter
    def param_trans_values(self, trans_values):
        for i, param in enumerate(self.params):
            param.trans_value = trans_values[i]

    @property
    def var_param_trans_values(self):
        return [var_param.trans_value for var_param in self.var_params]

    @var_param_trans_values.setter
    def var_param_trans_values(self, trans_values):
        for i, var_param in enumerate(self.var_params):
            var_param.trans_value = trans_values[i]

    def add_state_node(self,
                       *,
                       initial_mu,
                       initial_pi,
                       rho=0,
                       phi=0,
                       m=0,
                       omega=0):

        node = StateNode(initial_mu=initial_mu,
                         initial_pi=initial_pi,
                         rho=rho,
                         phi=phi,
                         m=m,
                         omega=omega)

        self._nodes.append(node)
        return node

    def add_binary_node(self):
        node = BinaryNode()
        self._nodes.append(node)
        return node

    def add_input_node(self, *, omega=0):
        node = InputNode(omega=omega)
        self._nodes.append(node)
        return node

    def add_binary_input_node(self, *, pihat=np.inf, eta0=0, eta1=1):
        node = BinaryInputNode(pihat=pihat, eta0=eta0, eta1=eta1)
        self._nodes.append(node)
        return node

    def reset(self):
        for input_node in self.input_nodes:
            input_node.reset_hierarchy()

    def undo_last_reset(self):
        for input_node in self.input_nodes:
            input_node.undo_last_reset_hierarchy()

    def recalculate(self):
        for input_node in self.input_nodes:
            input_node.recalculate()

    def surprise(self):
        surprise = 0
        for input_node in self.input_nodes:
            surprise += sum(input_node.surprises)
        return surprise

    def log_prior(self):
        if self.var_params:
            log_prior = 0
            for var_param in self.var_params:
                log_prior += var_param.log_prior()
            return log_prior
        else:
            raise ModelConfigurationError(
                'No variable (i.e., non-fixed) parameters.')

    def log_joint(self):
        return -self.surprise() + self.log_prior()

    def neg_log_joint_function(self):
        def f(trans_values):
            trans_values_backup = self.var_param_trans_values
            self.var_param_trans_values = trans_values
            try:
                self.recalculate()
                return -self.log_joint()
            except HgfUpdateError:
                self.var_param_trans_values = trans_values_backup
                return np.inf
        return f


# Standard 2-level HGF for continuous inputs
class StandardHGF(Model):
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
                 rho1=0,
                 rho2=0,
                 phi1=0,
                 m1=0,
                 phi2=0,
                 m2=0):

        # Superclass initialization
        super().__init__()

        # Set up nodes and their relationships
        self.x2 = self.add_state_node(initial_mu=initial_mu2,
                                      initial_pi=initial_pi2,
                                      omega=omega2,
                                      rho=rho2,
                                      phi=phi2,
                                      m=m2)
        self.x1 = self.add_state_node(initial_mu=initial_mu1,
                                      initial_pi=initial_pi1,
                                      omega=omega1,
                                      rho=rho1,
                                      phi=phi1,
                                      m=m1)
        self.xU = self.add_input_node(omega=omega_input)

        self.x1.add_volatility_parent(parent=self.x2, kappa=kappa1)
        self.xU.set_value_parent(parent=self.x1)

    # Input method
    def input(self, inputs):
        self.xU.input(inputs)


# Standard 3-level HGF for binary inputs
class StandardBinaryHGF(Model):
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
                 eta0=0,
                 eta1=1,
                 rho2=0,
                 rho3=0,
                 phi2=0,
                 m2=0,
                 phi3=0,
                 m3=0):

        # Superclass initialization
        super().__init__()

        # Set up nodes and their relationships
        self.x3 = self.add_state_node(initial_mu=initial_mu3,
                                      initial_pi=initial_pi3,
                                      omega=omega3,
                                      rho=rho3,
                                      phi=phi3,
                                      m=m3)
        self.x2 = self.add_state_node(initial_mu=initial_mu2,
                                      initial_pi=initial_pi2,
                                      omega=omega2,
                                      rho=rho2,
                                      phi=phi2,
                                      m=m2)
        self.x1 = self.add_binary_node()
        self.xU = self.add_binary_input_node(pihat=pihat_input,
                                             eta0=eta0,
                                             eta1=eta1)

        self.x2.add_volatility_parent(parent=self.x3, kappa=kappa2)
        self.x1.set_parent(parent=self.x2)
        self.xU.set_parent(parent=self.x1)

    # Input method
    def input(self, inputs):
        self.xU.input(inputs)


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


class Parameter(object):
    """Parameters of nodes"""
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

        if lower_bound is not None:
            self.lower_bound = lower_bound

        if upper_bound is not None:
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

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def lower_bound(self):
        return self._lower_bound

    @lower_bound.setter
    def lower_bound(self, lower_bound):
        space = self.space
        if lower_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "lower_bound must be None if space is 'native'.")
        elif lower_bound is not None and lower_bound > self.value:
            raise ParameterConfigurationError(
                'lower_bound may not be greater than current value')
        elif space is 'log':
            self._lower_bound = lower_bound
            self._upper_bound = None
        else:
            self._lower_bound = lower_bound

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def upper_bound(self):
        return self._upper_bound

    @upper_bound.setter
    def upper_bound(self, upper_bound):
        space = self.space
        if upper_bound is not None and space is 'native':
            raise ParameterConfigurationError(
                "upper_bound must be None if space is 'native'.")
        elif upper_bound is not None and upper_bound < self.value:
            raise ParameterConfigurationError(
                'upper_bound may not be less than current value')
        elif space is 'log':
            self._lower_bound = None
            self._upper_bound = upper_bound
        else:
            self._upper_bound = upper_bound

        # Recalculate trans_value
        try:
            self.value = self._value
        except AttributeError:
            pass
        # Recalculate trans_prior_mean
        try:
            self.prior_mean = self._prior_mean
        except AttributeError:
            pass

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if self.lower_bound is not None and value < self.lower_bound:
            raise ParameterConfigurationError(
                'value may not be less than current lower_bound')
        elif self.upper_bound is not None and value > self.upper_bound:
            raise ParameterConfigurationError(
                'value may not be greater than current upper_bound')
        else:
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

    def log_prior(self):
        try:
            return -gaussian_surprise(self.trans_value,
                                      self.trans_prior_mean,
                                      self.trans_prior_precision)
        except AttributeError as e:
            raise ModelConfigurationError(
                'trans_prior_mean and trans_prior_precision attributes ' +
                'must\nbe specified for method log_prior to return ' +
                'a value.') from e


def exp(x, *, lower_bound=0, upper_bound=None):
    """The (shifted and mirrored) exponential function"""
    if upper_bound is not None:
        return -np.exp(x) + upper_bound
    else:
        return np.exp(x) + lower_bound


def log(x, *, lower_bound=0, upper_bound=None):
    """The (shifted and mirrored) natural logarithm"""
    if upper_bound is not None:
        if x > upper_bound:
            raise LogArgumentError('Log argument may not be greater than ' +
                                   'upper bound.')
        elif x == upper_bound:
            return -np.inf
        else:
            return np.log(-x + upper_bound)
    else:
        if x < lower_bound:
            raise LogArgumentError('Log argument may not be less than ' +
                                   'lower bound.')
        elif x == lower_bound:
            return -np.inf
        else:
            return np.log(x - lower_bound)


def sgm(x, *, lower_bound=0, upper_bound=1):
    """The logistic sigmoid function"""
    return (upper_bound - lower_bound) / (1 + np.exp(-x)) + lower_bound


def logit(x, *, lower_bound=0, upper_bound=1):
    """The logistic function"""
    if x < lower_bound:
        raise LogitArgumentError('Logit argmument may not be less than ' +
                                 'lower bound.')
    elif x > upper_bound:
        raise LogitArgumentError('Logit argmument may not be greater than ' +
                                 'upper bound.')
    elif x == lower_bound:
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


class ModelConfigurationError(HgfException):
    """Model configuration error."""


class NodeConfigurationError(HgfException):
    """Node configuration error."""


class ParameterConfigurationError(HgfException):
    """Parameter configuration error."""


class HgfUpdateError(HgfException):
    """Error owing to a violation of the assumptions underlying HGF updates."""


class OutcomeValueError(HgfException):
    """Outcome value error."""


class LogArgumentError(HgfException):
    """Log argument out of bounds."""


class LogitArgumentError(HgfException):
    """Logit argument out of bounds."""
