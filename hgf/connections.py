import numpy as np

class Connection(object):
    """Connections between HGF nodes"""
    def __init__(self,
                 child,
                 parent):

    # Initialize attributes
        self.child = child
        self.parent = parent

    def send_bottom_up(self, message):
        parent = self.parent
        parent.receive(message, 'bottom_up')

    def send_top_down(self, message, flag):
        child = self.child
        return child.receive(message, flag)

    def send_posterior_top_down(self):
        pass

    def send_prediction_top_down(self):
        pass

# TODO: drop separate vape/vope signalling as first message!

class StateToStateValueConnection(Connection):
    """Connections for VAPE coupling between state nodes"""
    def __init__(self,
                 child,
                 parent,
                 psi=1):

    # Initialize attributes
        super().__init__(child, parent)
        self.psi = psi

    def send_bottom_up(self):
        child = self.child
        message = [self.psi**2 * child.pihats[-1] * child.vapes[-1], 
                   self.psi**2 * child.pihats[-1], 
                   child.times[-1]]
        super().send_bottom_up(message)

    def send_posterior_top_down(self):
        flag = 'top-down-value'
        parent = self.parent
        message = [self.psi * parent.mus[-1]]
        super().send_top_down(message, flag)


class InputToStateValueConnection(Connection):
    """Connections for VAPE coupling between input nodes and state nodes"""
    def __init__(self,
                 child,
                 parent):

    # Initialize attributes
        super().__init__(child, parent)

    def send_bottom_up(self):
        child = self.child
        message = [child.pihats[-1] * child.vapes[-1], 
                   child.pihats[-1], 
                   child.times[-1]]
        super().send_bottom_up(message)

    def send_posterior_top_down(self):
        flag = 'top-down-value'
        parent = self.parent
        message = [parent.mus[-1], 
                   parent.pis[-1]]
        super().send_top_down(message, flag)

    def send_prediction_top_down(self):
        flag = 'top-down-value'
        parent = self.parent
        message = [parent.muhats[-1]]
        super().send_top_down(message, flag)


class BinaryInputToBinaryConnection(Connection):
    """Connections for VAPE coupling between binary input nodes and binary nodes"""
    def __init__(self,
                 child,
                 parent):

    # Initialize attributes
        super().__init__(child, parent)

    def send_bottom_up(self):
        child = self.child

        if child.pihat == np.inf:
            message = [child.pihat.value, 
                       child.inputs[-1], 
                       child.times[-1]]
        else:
            message = [child.pihat.value, 
                       child.delta1s[-1], 
                       child.delta0s[-1], 
                       child.times[-1]]

        super().send_bottom_up(message)

    def send_prediction_top_down(self):
        flag = 'top-down-value'
        parent = self.parent
        message = [parent.muhats[-1]]
        super().send_top_down(message, flag)


class BinaryToStateConnection(Connection):
    """Connections for VAPE coupling between binary nodes and state nodes"""
    def __init__(self,
                 child,
                 parent):

    # Initialize attributes
        super().__init__(child, parent)

    def send_time_bottom_up(self, time):
        parent = self.parent
        message = parent.generate_new_prediction(time)
        return self.send_prediction_top_down()

    def send_bottom_up(self):
        child = self.child
        message = [child.vapes[-1], 
                   1 / child.pihats[-1], 
                   child.times[-1]]
        super().send_bottom_up(message)

    def send_prediction_top_down(self):
        flag = 'top-down-value'
        parent = self.parent
        message = [parent.muhats[-1]]
        return super().send_top_down(message, flag)

    def send_posterior_top_down(self):
        flag = 'top-down-post'
        message = None
        super().send_top_down(message, flag)


class StateToStateVolatilityConnection(object):
    """Connections for VOPE coupling between nodes"""
    def __init__(self,
                 child,
                 parent,
                 kappa=1):

    # Initialize attributes
        super().__init__(child, parent)
        self.kappa = kappa
        self.type = 'volatility'

    def send_bottom_up(self):
        child = self.child
        message = [self.kappa * child.gammas[-1] * child.vopes[-1], 
                   1 / 2 * self.kappa**2 * child.gammas[-1]**2 *\
                   (1 + 2 * child.vopes[-1] - child.vopes[-1] / child.gammas[-1]),
                   child.times[-1]]
        super().send_bottom_up(message)

    def send_posterior_top_down(self):
        flag = 'top-down-volatility'
        parent = self.parent
        message = [self.kappa * parent.mus[-1]]
        super().send_top_down(message, flag)


# We could also drop this and use the regular state to state connection
class InputToStateVolatilityConnection(Connection):
    """Connections for VOPE coupling between input nodes and state nodes"""
    def __init__(self,
                 child,
                 parent):

    # Initialize attributes
        super().__init__(child, parent)

    def send_bottom_up(self):
        child = self.child
        message = [1 / 2 * self.kappa * child.vopes[-1],
                   1 / 2 * self.kappa **2 * (1 + child.vopes[-1]),
                   child.times[-1]]
        super().send_bottom_up(message)

    def send_posterior_top_down(self):
        flag = 'top-down-volatility'
        parent = self.parent
        message = [self.kappa * parent.mus[-1]]
        super().send_top_down(message, flag)

