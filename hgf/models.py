import numpy as np
from hgf.nodes import *
from hgf.connections import *
from hgf.exceptions import ModelConfigurationError, HgfUpdateError

# Generic HGF model
class Model(object):
    """Generic HGF model"""
    def __init__(self):
        self._nodes = []
        self._connections = []

    @property
    def nodes(self):
        return self._nodes

    @property
    def connections(self):
        return self._connections

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
        for connection in self.connections:
            params.extend(connection.params)
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

    def add_value_connection(self, child, parent, psi=1):
        if isinstance(parent, StateNode):
            if isinstance(child, StateNode):
                c = StateToStateValueConnection(child, parent, psi)
            elif isinstance(child, InputNode):
                c = InputToStateValueConnection(child, parent)
            elif isinstance(child, BinaryNode):
                c = BinaryToStateConnection(child, parent)
            else:
                raise ModelConfigurationError(
                'State nodes can only be value parents to state nodes, binary nodes, and continuous input nodes.')
        elif isinstance(parent, BinaryNode):
            if isinstance(child, BinaryInputNode):
                c = BinaryInputToBinaryConnection(child, parent)
            else: 
                raise ModelConfigurationError(
                'Binary nodes can only be parents to binary input nodes.')
        else:
            raise ModelConfigurationError(
            'Only state nodes and binary nodes can be value parents.')
        self._connections.append(c)

    def add_volatility_connection(self, child, parent, kappa=1):
        if isinstance(parent, StateNode):
            if isinstance(child, StateNode):
                c = StateToStateVolatilityConnection(child, parent, kappa)
            elif isinstance(child, InputNode):
                c = InputToStateVolatilityConnection(child, parent, kappa)
            else:
                raise ModelConfigurationError(
                'State nodes can only be volatility parents to continuous input nodes and state nodes.')
        else:
            raise ModelConfigurationError(
            'Only state nodes can be volatility parents to other nodes.')
        self._connections.append(c)

    def reset(self):
        for node in self.nodes:
            node.reset()

    def undo_last_reset(self):
        for node in self.nodes:
            node.undo_last_reset()

    # TODO: make this work for multiple input nodes
    def recalculate(self):
        for input_node in self.input_nodes:
            iwt = list(input_node.inputs_with_times[1:])
        for node in self.nodes:
            node.reset()
        for input_node in self.input_nodes:
            try:
                input_node.input(iwt)
            except HgfUpdateError as e:
                for node in self.nodes:
                    node.undo_last_reset()
                raise e

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

        self.add_volatility_connection(child=self.x1, parent=self.x2, kappa=kappa1)
        self.add_value_connection(child=self.xU, parent=self.x1)

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

        self.add_volatility_connection(child=self.x2, parent=self.x3, kappa=kappa2)
        self.add_value_connection(child=self.x1, parent=self.x2)
        self.add_value_connection(child=self.xU, parent=self.x1)

    # Input method
    def input(self, inputs):
        self.xU.input(inputs)


