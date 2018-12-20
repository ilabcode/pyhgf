import numpy as np
from hgf.connections import *
from hgf.parameters import *
from hgf.utils import *
from hgf.exceptions import NodeConfigurationError, HgfUpdateError


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

        # Initialize connections
        self.td_con = []
        self.bo_cons = []

        # Initialize time series
        self.times = [0]
        self.pihats = [None]
        self.pis = [self.initial_pi.value]
        self.muhats = [None]
        self.mus = [self.initial_mu.value]
        self.gammas = [None]
        self.vapes = [None]
        self.vopes = [None]
        self.driftrate = self.rho.value
        self.logvol = self.omega.value

    @property
    def connections(self):
        connections = []
        connections.extend(self.td_con)
        connections.extend(self.bo_cons)
        return connections

    @property
    def params(self):
        params = [self.initial_mu,
                  self.initial_pi,
                  self.rho,
                  self.phi,
                  self.m,
                  self.omega]

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

        self._gammas_backup = self.gammas
        self.gammas = [None]

        self._vapes_backup = self.vapes
        self.vapes = [None]

        self._vopes_backup = self.vopes
        self.vopes = [None]

        self._driftrate_backup = self.driftrate
        self.driftrate = self.rho.value

        self._logvol_backup = self.logvol
        self.logvol = self.omega.value


    def undo_last_reset(self):
        self.times = self._times_backup
        self.pihats = self._pihats_backup
        self.pis = self._pis_backup
        self.muhats = self._muhats_backup
        self.mus = self._mus_backup
        self.nus = self._nus_backup
        self.gammas = self._gammas_backup
        self.vapes = self._vapes_backup
        self.vopes = self._vopes_backup
        self.driftrate = self._driftrate_backup
        self.logvol = self._logvol_backup

    # TODO
    #def reset_hierarchy(self):
    #    self.reset()
    #    for pa in self.parents:
    #        pa.reset_hierarchy()

    #def undo_last_reset_hierarchy(self):
    #    self.undo_last_reset()
    #    for pa in self.parents:
    #        pa.undo_last_reset_hierarchy()

    def set_top_down_connection(self, tdcon):
        self.td_con.append(tdcon)

    def add_bottom_up_connection(self, bocon):
        self.bo_cons.append(bocon)

    def send_bottom_up(self):
        for i, bocon in self.bo_cons:
            self.bo_cons[i].send_bottom_up()

    def send_posterior_top_down(self):
        self.td_con.send_posterior_top_down()

    def send_prediction_top_down(self):
        self.td_con.send_prediction_top_down()

    def receive(self, message, flag):
        if flag == 'bottom-up':
            self.update(self, message)
        elif flag == 'top-down-value':
            self.driftrate += message[0]
        elif flag == 'top-down-volatility':
            self.logvol += message[0]

    def update(self, message):
        # compute most recent prediction
        time = message[-1]
        self.predict_current(self, time)
        
        # compute new posteriors and store them
        pihat = self.pihats[-1]
        muhat = self.muhats[-1]

        pi = pihat + message[1]
        mu = muhat + (message[0] / pi)

        self.pis.append(pi)
        self.mus.append(mu)
        self.times.append(time)

        # reset driftrate and logvol 
        self.driftrate = self.rho.value
        self.logvol = self.omega.value

        # compute prediction errors
        self.prediction_error()

        # signal posteriors top-down
        for i, td_con in enumerate(self.td_cons):
            self.td_cons[i].send_posterior_top_down()

    def predict_current(self, time):
        # generate current predictions and store them
        t = time - self.times[-1]
        muhat = self.new_muhat(t)
        nu    = self.new_nu(t)
        pihat = self.new_pihat(nu)
        gamma = self.new_gamma(pihat, nu)

        self.muhats.append(muhat)
        self.pihats.append(pihat)
        self.gammas.append(gamma)

        # signal prediction top-down if needed
        for i, td_con in enumerate(self.td_cons):
            self.td_cons[i].send_prediction_top_down()
                                                                     
    def new_muhat(self, t):
        return self.mus[-1] + t * self.driftrate

    def new_nu(self, t):
        nu = t * np.exp(self.logvol)
        if nu > 1e-128:
            return nu
        else:
            raise HgfUpdateError(
                'Nu is zero. Parameters values are in region where model\n' +
                'assumptions are violated.')

    def new_pihat(self, nu):
        return 1 / (1 / self.pis[-1] + nu)

    def new_gamma(pihat, nu):
        return pihat * nu

    def prediction_error(self):
        # compute prediction errors and store them
        vape = self.vape()
        vope = self.vope(vape)
    
        self.vapes.append(vape)
        self.vopes.append(vope)

        # send weighted prediction errors bottom-up
        self.send_bottom_up()

    def vape(self):
        return self.mus[-1] - self.muhats[-1]

    def vope(self, vape):
        return self.pihats[-1] * (1 / self.pis[-1] + vape**2) -1

    # this is only needed for communication with binary nodes, which can 
    # prompt state nodes to compute new predictions of their mean, given 
    # current time
    def generate_new_prediction(self, time):
        t = time - self.times[-1]
        muhat = self.new_muhat(t)
        return muhat

# HGF binary state node
class BinaryNode(object):
    """HGF binary state node"""
    def __init__(self):

        # Initialize connections
        self.td_con = None
        self.bo_con = None

        # Initialize time series
        self.times = [0]
        self.pihats = [None]
        self.pis = [None]
        self.muhats = [None]
        self.mus = [None]
        self.vapes = [None]

    @property
    def connections(self):
        connections = []
        if self.td_con:
            connections.append(self.td_con)
        if self.bo_con:
            connections.append(self.bo_con)
        return connections

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

        self._vapes_backup = self.vapes
        self.vapes = [None]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.pihats = self._pihats_backup
        self.pis = self._pis_backup
        self.muhats = self._muhats_backup
        self.mus = self._mus_backup
        self.vapes = self._vapes_backup

    #TODO
    #def reset_hierarchy(self):
    #    self.reset()
    #    for pa in self.parents:
    #        pa.reset_hierarchy()

    #def undo_last_reset_hierarchy(self):
    #    self.undo_last_reset()
    #    for pa in self.parents:
    #        pa.undo_last_reset_hierarchy()

    def set_bottom_up_connection(self, bocon):
        self.bo_con = bocon

    def set_top_down_connection(self, tdcon):
        self.td_con = tdcon

    def send_bottom_up(self):
        self.bo_con.send_bottom_up()

    def send_prediction_top_down(self):
        self.td_con.send_prediction_top_down()

    def receive(self, message, flag):
        if flag == 'bottom-up':
            self.update(message)
        elif flag == 'top-down-value':
            return message
        elif flag == 'top-down-post':
            self.send_prediction_top_down()

    def prompt_parent_prediction(self, time):
        message = self.bocon.send_time_bottom_up(time)
        return message

    def update(self, message):
        if message[0] == inf:
            pi = message[0]
            mu = message[1]
            time = message[2]
        else: 
            pihat_inp = message[0]
            delta1_inp = message[1]
            delta0_inp = message[2]
            time = message[3]

            muhat_pa = self.prompt_parent_prediction(time)
            self.compute_prediction(muhat_pa)
            
            lik1 = self.muhat[-1] * np.exp(-1 / 2 * pihat_inp * delta1_inp**2)
            lik2 = (1 - self.muhat[-1]) * np.exp(-1 / 2 * pihat_inp * delta0_inp**2)
            
            mu = lik1 / (lik1 + lik2)
            pi = 1 / (mu * (1 - mu))
            time = message[3]
        
        self.mus.append(mu)
        self.pis.append(pi)
        self.times.append(time)

        self.prediction_error()

    def compute_prediction(self, muhat_pa):
        muhat = 1 / (1 + np.exp(- muhat_pa))
        pihat = 1 / (muhat * (1 - muhat))

        self.muhats.append(muhat)
        self.pihats.append(pihat)

    def prediction_error(self):
        vape = self.vape
        self.vapes.append(vape)
        self.send_bottom_up(self.bocon)

    def vape(self):
        return self.mus[-1] - self.muhats[-1]

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

        # Initialize connections
        self.bo_con = None

        # Initialize time series
        self.times = [0]
        self.inputs = [None]
        self.inputs_with_times = [(None, 0)]
        self.delta1s = [None]
        self.delta0s = [None]
        self.surprises = [0]

    @property
    def connections(self):
        connections = []
        if self.bo_con is not None:
            connections.append(self.bo_con)
        return connections

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

        self._delta1s_backup = self.delta1s
        self.delta1s = [None]

        self._delta0s_backup = self.delta0s
        self.delta0s = [None]

        self._surprises_backup = self.surprises
        self.surprises = [0]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.inputs = self._inputs_backup
        self.inputs_with_times = self._inputs_with_times_backup
        self.delta1s = self._delta1s_backup
        self.delta0s = self._delta0s_backup
        self.surprises = self._surprises_backup

    #TODO
    #def reset_hierarchy(self):
    #    self.reset()
    #    for pa in self.parents:
    #        pa.reset_hierarchy()

    #def undo_last_reset_hierarchy(self):
    #    self.undo_last_reset()
    #    for pa in self.parents:
    #        pa.undo_last_reset_hierarchy()

    #def recalculate(self):
    #    iwt = list(self.inputs_with_times[1:])
    #    self.reset_hierarchy()
    #    try:
    #        self.input(iwt)
    #    except HgfUpdateError as e:
    #        self.undo_last_reset_hierarchy()
    #        raise e

    def set_bottom_up_connection(self, bocon):
        self.bo_con = bocon

    def send_bottom_up(self):
        self.bo_con.send_bottom_up()

    def receive(self, message, flag):
        if flag == 'top-down-value':
            self.compute_surprise(message)

    def input(self, inputs):
        try:
            for input in inputs:
                try:
                    value = input[0]
                    time = input[1]
                except IndexError:
                    value = input[0]
                    time = self.times[-1] + 1
                finally:
                    self.receive_single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self.receive_single_input(value, time)

    def receive_single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.inputs_with_times.append((value, time))

        if self.pihat != np.inf:
            self.prediction_error()

        self.send_bottom_up()

    def prediction_error(self):
        delta1 = self.inputs[-1] - self.eta1.value
        delta0 = self.inputs[-1] - self.eta0.value

        self.delta1s.append(delta1)
        self.delta0s.append(delta0)

    def compute_surprise(self, muhat_pa):
        pihat = self.pihat.value
        value = self.inputs[-1]
        if pihat == np.inf:
            surprise = binary_surprise(value, muhat_pa)
        else:
            eta1 = self.eta1.value
            eta0 = self.eta0.value
            surprise = (-np.log(muhat_pa * gaussian(value, eta1, pihat) +
                        (1 - muhat_pa) * gaussian(value, eta0, pihat)))
        self.surprises.append(surprise)
        
                            
