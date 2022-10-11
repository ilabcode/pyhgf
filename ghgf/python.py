"""The HGF time series model."""

from math import exp, log, sqrt
from typing import Dict, List, Optional

import numpy as np
from numba import jit


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
            if isinstance(node, InputNode) or isinstance(node, BinaryInputNode):
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

    def add_state_node(
        self,
        *,
        initial_mu: float,
        initial_pi: float,
        rho: float = 0.0,
        phi: float = 0.0,
        m: float = 0.0,
        omega: float = 0.0,
    ):

        node = StateNode(
            initial_mu=initial_mu,
            initial_pi=initial_pi,
            rho=rho,
            phi=phi,
            m=m,
            omega=omega,
        )

        self._nodes.append(node)
        return node

    def add_binary_node(self):
        node = BinaryNode()
        self._nodes.append(node)
        return node

    def add_input_node(self, *, omega: float = 0):
        node = InputNode(omega=omega)
        self._nodes.append(node)
        return node

    def add_binary_input_node(
        self, *, pihat: float = np.inf, eta0: float = 0, eta1: float = 1
    ):
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
            raise ModelConfigurationError("No variable (i.e., non-fixed) parameters.")

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
    """The standard n-level HGF for continuous inputs.

    The standard continuous HGF can implements the Gaussian Random Walk and AR1
    perceptual models.

    Parameters
    ----------
    n_levels : int
        The number of hierarchies in the perceptual model (default sets to `2`).
    process_type : str or None
        The model type to use (can be "GRW" or "AR1"). If `process_type` is not
        provided, it is infered from the parameters provided. If both `phi` and `rho`
        are None or dictionary, an error is returned.
    initial_mu : dict
        Dictionary containing the initial values for the `initial_mu` parameter at
        different levels of the hierarchy. Defaults set to `{"1": 0.0, "2": 0.0}`
        for a 2-levels model.
    initial_pi : dict
        Dictionary containing the initial values for the `initial_pi` parameter at
        different levels of the hierarchy. Pis values encode the precision of the
        values at each level (Var = 1/pi) Defaults set to `{"1": 1.0, "2": 1.0}` for
        a 2-levels model.
    omega : dict
        Dictionary containing the initial values for the `omega` parameter at
        different levels of the hierarchy. Omegas represent the tonic part of the
        variance (the part that is not affected by the parent node). Defaults set to
        `{"1": -10.0, "2": -10.0}` for a 2-levels model. This parameters only when
        `process_type="GRW"`.
    omega_input: float
        Default value sets to `np.log(1e-4)`. Represent the noise associated with the
        input.
    rho : dict or None
        Dictionary containing the initial values for the `rho` parameter at
        different levels of the hierarchy. Rho represents the drift of the random walk.
        Only required when `process_type="GRW"`. Defaults set all entries to `0`
        according to the number of required levels.
    kappa : dict
        Dictionary containing the initial values for the `kappa` parameter at
        different levels of the hierarchy. Kappa represents the phasic part of the
        variance (the part that is affected by the parents nodes) and will defines the
        strenght of the connection between the nod eand the parent node. Often fixed
        to 1. Defaults set to `{"1": 1.0}` for a 2-levels model. Only required when
        `process_type="GRW"`.
    phi : dict or None
        Dictionary containing the initial values for the `phi` parameter at
        different levels of the hierarchy. Phi should always be between 0 and 1.
        Defaults set all entries to `0` according to the number of required levels.
        `phi` is only required when `process_type="AR1"`.
    m : dict or None
        Dictionary containing the initial values for the `m` parameter at
        different levels of the hierarchy. Defaults set all entries to `0` according to
        the number of required levels. `m` is only required when `process_type="AR1"`.
    verbose : bool
        Default is `True` (show bar progress).

    Attributes
    ----------
    n_levels : int
        The number of hierarchies in the model. Cannot be less than 2.
    process_type : str
        The model implemented (can be `"AR1"` or `"GRW"`).

    Notes
    -----
    The model used by the perceptual model is defined by the `process_type` parameter
    (can be `"GRW"` or `"AR1"`). If `process_type` is not provided, the class will try
    to determine it automatically by looking at the `rho` and `phi` parameters. If
    `rho` is provided `process_type="GRW"`, if `phi` is provided `process_type="AR1"`.
    If both `phi` and `rho` are `None` an error will be returned.

    """

    def __init__(
        self,
        n_levels: int = 2,
        process_type: Optional[str] = None,
        initial_mu: Dict[str, float] = {"1": 0.0, "2": 0.0},
        initial_pi: Dict[str, float] = {"1": 1.0, "2": 1.0},
        omega_input: float = np.log(1e-4),
        omega: Dict[str, float] = {"1": -10.0, "2": -10.0},
        kappas: Dict[str, float] = {"1": 1.0},
        rho: Optional[Dict[str, float]] = None,
        phi: Optional[Dict[str, float]] = None,
        m: Dict[str, float] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        print("Continuous Hierarchical Gaussian Filter")
        self.n_levels = n_levels

        # Determine which perceptual model to use
        if (phi is not None) and (rho is not None) & (process_type is None):
            raise ValueError(
                (
                    "Unable to determine the model type to use. "
                    "Provide process_type or one of the rho or phi parameters."
                )
            )
        elif isinstance(process_type, str):
            if (process_type == "GRW") and (rho is None):
                rho = {}
                for i in range(n_levels):
                    rho[str(i + 1)] = 0.0
            elif (process_type == "AR1") and (phi is None) and (m is None):
                phi, m = {}, {}
                for i in range(n_levels):
                    phi[str(i + 1)] = 0.0
                    m[str(i + 1)] = 0.0
            elif process_type not in ["GRW", "AR1"]:
                raise ValueError(
                    ("Invalid model type specification. Should be 'AR1' or 'GRW'.")
                )
        else:
            if (phi is None) and (rho is None):
                raise ValueError(
                    (
                        "The parameters phi and rho are both None."
                        "Please provide at least one of them."
                    )
                )
            elif phi is None:
                process_type = "GRW"
            elif rho is None:
                process_type = "AR1"

        self.process_type = process_type

        # Sanity checks - Make sure that the parameters declared match with the
        # number of levels required
        if n_levels < 2:
            raise ValueError("The number of levels cannot be less than 2")
        if self.process_type == "GRW":
            params = [initial_mu, initial_pi, omega, rho]
            names = ["initial_mu", "initial_pi", "omega", "rho"]
        else:
            params = [initial_mu, initial_pi, omega, phi, m]
            names = ["initial_mu", "initial_pi", "omega", "phi", "m"]
        for param, name in zip(params, names):
            if len(param) != n_levels:  # type: ignore
                raise ValueError(
                    (
                        f"The size of {name} is {len(param)}",  # type: ignore
                        f" and does not match with the number of levels ({n_levels})",
                    )
                )
        if len(kappas) != n_levels - 1:
            raise ValueError(
                (
                    "The size of kappa does not match with"
                    f"the number of levels (should be {n_levels-1})",
                )
            )

        # Superclass initialization
        super().__init__()

        # Set up nodes - Loop across the number of levels required and set values
        # and volatility parents accordingly
        print(
            f"... Initializing a {self.n_levels} levels perceptual HGF "
            f"using a {self.process_type} model."
        )
        for n in range(n_levels, 0, -1):
            if self.process_type == "GRW":
                setattr(
                    self,
                    f"x{n}",
                    self.add_state_node(
                        initial_mu=initial_mu[str(n)],
                        initial_pi=initial_pi[str(n)],
                        omega=omega[str(n)],
                        rho=rho[str(n)],  # type: ignore
                    ),
                )
            elif self.process_type == "AR1":
                setattr(
                    self,
                    f"x{n}",
                    self.add_state_node(
                        initial_mu=initial_mu[str(n)],
                        initial_pi=initial_pi[str(n)],
                        omega=omega[str(n)],
                        phi=phi[str(n)],  # type: ignore
                        m=m[str(n)],  # type: ignore
                    ),
                )

        self.xU = self.add_input_node(omega=omega_input)

        # Set up nodes relationships
        for n in range(1, n_levels):
            getattr(self, f"x{n}").add_volatility_parent(
                parent=getattr(self, f"x{n+1}"), kappa=kappas[str(n)]
            )

        self.xU.set_value_parent(parent=self.x1)  # type: ignore

    # Input method
    def input(self, inputs):
        self.xU.input(inputs)


# Standard 3-level HGF for binary inputs
class StandardBinaryHGF(Model):
    """The standard 3-level HGF for binary inputs

    Parameters
    ----------

    """

    def __init__(
        self,
        *,
        initial_mu2: float,
        initial_pi2: float,
        initial_mu3: float,
        initial_pi3: float,
        omega2: float,
        kappa2: float,
        omega3: float,
        pihat_input: float = np.inf,
        eta0: float = 0,
        eta1: float = 1,
        rho2: float = 0,
        rho3: float = 0,
        phi2: float = 0,
        m2: float = 0,
        phi3: float = 0,
        m3: float = 0,
    ):

        # Superclass initialization
        super().__init__()

        # Set up nodes and their relationships
        self.x3 = self.add_state_node(
            initial_mu=initial_mu3,
            initial_pi=initial_pi3,
            omega=omega3,
            rho=rho3,
            phi=phi3,
            m=m3,
        )
        self.x2 = self.add_state_node(
            initial_mu=initial_mu2,
            initial_pi=initial_pi2,
            omega=omega2,
            rho=rho2,
            phi=phi2,
            m=m2,
        )
        self.x1 = self.add_binary_node()
        self.xU = self.add_binary_input_node(pihat=pihat_input, eta0=eta0, eta1=eta1)

        self.x2.add_volatility_parent(parent=self.x3, kappa=kappa2)
        self.x1.set_parent(parent=self.x2)
        self.xU.set_parent(parent=self.x1)

    # Input method
    def input(self, inputs):
        self.xU.input(inputs)


# HGF continuous state node
class StateNode(object):
    """HGF continuous state node"""

    def __init__(self, *, initial_mu, initial_pi, rho=0, phi=0, m=0, omega=0):

        # Sanity check
        if rho and phi:
            raise NodeConfigurationError(
                "hgf.StateNode: rho (drift) and phi (AR(1) parameter) may "
                + "not be non-zero at the same time."
            )

        # Initialize parameter attributes
        self.initial_mu = Parameter(value=initial_mu)
        self.initial_pi = Parameter(value=initial_pi, space="log")
        self.rho = Parameter(value=rho)
        self.phi = Parameter(value=phi, space="logit")
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
        params = [
            self.initial_mu,
            self.initial_pi,
            self.rho,
            self.phi,
            self.m,
            self.omega,
        ]

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
        self.kappas.append(Parameter(value=kappa, space="log"))

    def new_muhat(self, time):
        t = time - self.times[-1]
        driftrate = self.rho.value
        for i, _ in enumerate(self.va_pas):
            driftrate += self.psis[i].value * self.va_pas[i].mus[-1]
        return self.mus[-1] + t * driftrate

    def _new_nu(self, time):
        t = time - self.times[-1]
        logvol = self.omega.value
        for i, _ in enumerate(self.vo_pas):
            logvol += self.kappas[i].value * self.vo_pas[i].mus[-1]
        nu = t * exp(logvol)
        if nu > 1e-128:
            return nu
        else:
            raise HgfUpdateError(
                "Nu is zero. Parameters values are in region where model\n"
                + "assumptions are violated."
            )

    def new_pihat_nu(self, time):
        new_nu = self._new_nu(time)
        return [1 / (1 / self.pis[-1] + new_nu), new_nu]

    def vape(self):
        return self.mus[-1] - self.muhats[-1]

    def vope(self):
        return (1 / self.pis[-1] + self.vape() ** 2) * self.pihats[-1] - 1

    @staticmethod
    @jit(nopython=True)
    def numba_update_value_parent(
        i: int, va_pa: List, time: List, psis: List, pihat: List, vape: List
    ):
        pihat_pa, nu_pa = va_pa.new_pihat_nu(time)  # type: ignore
        pi_pa = pihat_pa + psis[i].value ** 2 * pihat
        muhat_pa = va_pa.new_muhat(time)  # type: ignore
        mu_pa = muhat_pa + psis[i].value * pihat / pi_pa * vape

        return pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa

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
            pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa = self.numba_update_value_parent(
                i, va_pa, time, psis, pihat, vape
            )
            va_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

        # Update volatility parents
        nu = self.nus[-1]
        kappas = self.kappas
        vope = self.vope()

        for i, vo_pa in enumerate(vo_pas):
            pihat_pa, nu_pa = vo_pa.new_pihat_nu(time)
            pi_pa = pihat_pa + 0.5 * (kappas[i].value * nu * pihat) ** 2 * (
                1 + (1 - 1 / (nu * self.pis[-2])) * vope
            )
            if pi_pa <= 0:
                raise HgfUpdateError(
                    "Negative posterior precision. Parameters values are\n"
                    + "in a region where model assumptions are violated."
                )

            muhat_pa = vo_pa.new_muhat(time)
            mu_pa = muhat_pa + 0.5 * kappas[i].value * nu * pihat / pi_pa * vope

            vo_pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa, nu_pa)

    def update(self, time, pihat, pi, muhat, mu, nu):
        self.times.append(time)
        self.pihats.append(pihat)
        self.pis.append(pi)
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

        self.inputs_backup = self.inputs
        self.inputs = [None]

        self.inputs_with_times_backup = self.inputs_with_times
        self.inputs_with_times = [(None, 0)]

        self._surprises_backup = self.surprises
        self.surprises = [0]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.inputs = self.inputs_backup
        self.inputs_with_times = self.inputs_with_times_backup
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
            self.input(iwt, verbose=False)
        except HgfUpdateError as e:
            self.undo_last_reset_hierarchy()
            raise e

    def set_value_parent(self, *, parent):
        self.va_pa = parent

    def set_volatility_parent(self, *, parent, kappa):
        self.vo_pa = parent
        self.kappa = Parameter(value=kappa, space="log")

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

        pihat = 1 / exp(lognoise)

        # Update value parent
        pihat_va_pa, nu_va_pa = va_pa.new_pihat_nu(time)
        pi_va_pa = pihat_va_pa + pihat

        muhat_va_pa = va_pa.new_muhat(time)
        vape = value - muhat_va_pa
        mu_va_pa = muhat_va_pa + pihat / pi_va_pa * vape

        va_pa.update(time, pihat_va_pa, pi_va_pa, muhat_va_pa, mu_va_pa, nu_va_pa)

        # Update volatility parent
        if vo_pa is not None:
            vope = (1 / pi_va_pa + (value - mu_va_pa) ** 2) * pihat - 1

            pihat_vo_pa, nu_vo_pa = vo_pa.new_pihat_nu(time)
            pi_vo_pa = pihat_vo_pa + 0.5 * kappa**2 * (1 + vope)
            if pi_vo_pa <= 0:
                raise HgfUpdateError(
                    "Negative posterior precision. Parameters values are\n"
                    + "in a region where model assumptions are violated."
                )

            muhat_vo_pa = vo_pa.new_muhat(time)
            mu_vo_pa = muhat_vo_pa + 0.5 * kappa / pi_vo_pa * vope

            vo_pa.update(time, pihat_vo_pa, pi_vo_pa, muhat_vo_pa, mu_vo_pa, nu_vo_pa)

        return gaussian_surprise(value, muhat_va_pa, pihat)

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.inputs_with_times.append((value, time))
        self.surprises.append(self.update_parents(value, time))

    def input(self, inputs, verbose=True):
        """Add data to the input node.

        Parameters
        ----------
        inputs : list or np.ndarray
            The input time series.
        verbose : bool
            If `True`, show the progress bar.

        """
        try:
            for this_input in inputs:
                try:
                    value = this_input[0]
                    time = this_input[1]
                except IndexError:
                    value = this_input
                    time = self.times[-1] + 1
                finally:
                    self._single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self._single_input(value, time)


class BinaryInputNode(object):
    """An HGF node that receives binary input.

    Parameters
    ----------
    pihat : float
    eta0 : float
    eta1 : float

    Attributes
    ----------
    pihat
    eta0
    eta1
    pa : None
        Parent nodes.
    times : list
    iputs : list
    inputs_with_time : list
    surprises : list

    """

    def __init__(self, *, pihat: float = np.inf, eta0: float = 0.0, eta1: float = 1.0):

        # Incorporate parameter attributes
        self.pihat = Parameter(value=pihat, space="log")
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
        return [self.pihat, self.eta0, self.eta1]

    def reset(self):
        self._times_backup = self.times
        self.times = [0]

        self.inputs_backup = self.inputs
        self.inputs = [None]

        self.inputs_with_times_backup = self.inputs_with_times
        self.inputs_with_times = [(None, 0)]

        self._surprises_backup = self.surprises
        self.surprises = [0]

    def undo_last_reset(self):
        self.times = self._times_backup
        self.inputs = self.inputs_backup
        self.inputs_with_times = self.inputs_with_times_backup
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
            und1 = exp(-pihat / 2 * (value - eta1) ** 2)
            # Likelihood under eta0
            und0 = exp(-pihat / 2 * (value - eta0) ** 2)
            # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
            mu_pa = muhat_pa * und1 / (muhat_pa * und1 + (1 - muhat_pa) * und0)
            pi_pa = 1 / (mu_pa * (1 - mu_pa))
            # Surprise
            surprise = -log(
                muhat_pa * gaussian(value, eta1, pihat)
                + (1 - muhat_pa) * gaussian(value, eta0, pihat)
            )

        pa.update(time, pihat_pa, pi_pa, muhat_pa, mu_pa)

        return surprise

    def _single_input(self, value, time):
        self.times.append(time)
        self.inputs.append(value)
        self.inputs_with_times.append((value, time))
        self.surprises.append(self.update_parent(value, time))

    def input(self, inputs):
        try:
            for this_input in inputs:
                try:
                    value = this_input[0]
                    time = this_input[1]
                except IndexError:
                    value = this_input
                    time = self.times[-1] + 1
                finally:
                    self._single_input(value, time)
        except TypeError:
            value = inputs
            time = self.times[-1] + 1
            self._single_input(value, time)


class Parameter(object):
    """Parameters of nodes.

    Parameters
    ----------
    space: string
        Default sets to `"native"`.
    lower_bound : float or None
        Default sets to `None`.
    upper_bound : float or None
        Default sets to `None`.
    value : float or None
        Default sets to `None`.
    trans_value : float or None
        Default sets to `None`.
    prior_mean : float or None
        Default sets to `None`.
    trans_prior_mean : float or None
        Default sets to `None`.
    trans_prior_precision : float or None
        Default sets to `None`.

    """

    def __init__(
        self,
        *,
        space: str = "native",
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        value: Optional[float] = None,
        trans_value: Optional[float] = None,
        prior_mean: Optional[float] = None,
        trans_prior_mean: Optional[float] = None,
        trans_prior_precision: Optional[float] = None,
    ):

        # Initialize attributes
        self.space = space

        if lower_bound is not None:
            self.lower_bound = lower_bound

        if upper_bound is not None:
            self.upper_bound = upper_bound

        if value is not None and trans_value is not None:
            raise ParameterConfigurationError(
                "Only one of value and trans_value can be given."
            )
        elif value is not None:
            self.value = value
        elif trans_value is not None:
            self.trans_value = trans_value
        else:
            raise ParameterConfigurationError(
                "One of value and trans_value must be given."
            )

        if prior_mean is not None and trans_prior_mean is not None:
            raise ParameterConfigurationError(
                "Only one of prior_mean and trans_prior_mean can be given."
            )
        elif prior_mean is not None:
            self.prior_mean = prior_mean
        else:
            self.trans_prior_mean = trans_prior_mean

        if trans_prior_precision is None and (
            prior_mean is not None or trans_prior_mean is not None
        ):
            raise ParameterConfigurationError(
                "trans_prior_precision must be given if prior_mean "
                + "or trans_prior_mean is given"
            )
        else:
            self.trans_prior_precision = trans_prior_precision

    @property
    def space(self):
        return self._space

    @space.setter
    def space(self, space: str):
        if space == "native":
            self._space = space
            self._lower_bound = None
            self._upper_bound = None
        elif space == "log":
            self._space = space
            self._lower_bound = 0
            self._upper_bound = None
        elif space == "logit":
            self._space = space
            self._lower_bound = 0
            self._upper_bound = 1
        else:
            raise ParameterConfigurationError(
                "Space must be one of 'native, 'log', or 'logit'"
            )

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
    def lower_bound(self, lower_bound: Optional[int]):
        space = self.space
        if lower_bound is not None and space == "native":
            raise ParameterConfigurationError(
                "lower_bound must be None if space is 'native'."
            )
        elif lower_bound is not None and lower_bound > self.value:
            raise ParameterConfigurationError(
                "lower_bound may not be greater than current value"
            )
        elif space == "log":
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
    def upper_bound(self, upper_bound: Optional[int]):
        space = self.space
        if upper_bound is not None and space == "native":
            raise ParameterConfigurationError(
                "upper_bound must be None if space is 'native'."
            )
        elif upper_bound is not None and upper_bound < self.value:
            raise ParameterConfigurationError(
                "upper_bound may not be less than current value"
            )
        elif space == "log":
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
    def value(self, value: Optional[float]):
        if self.lower_bound is not None and value < self.lower_bound:
            raise ParameterConfigurationError(
                "value may not be less than current lower_bound"
            )
        elif self.upper_bound is not None and value > self.upper_bound:
            raise ParameterConfigurationError(
                "value may not be greater than current upper_bound"
            )
        else:
            self._value = value

        space = self.space
        if space == "native":
            self._trans_value = value
        elif space == "log":
            self._trans_value = log_shift_mirror(
                value, lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )
        elif space == "logit":
            self._trans_value = logit(
                value, lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )

    @property
    def trans_value(self):
        return self._trans_value

    @trans_value.setter
    def trans_value(self, trans_value: Optional[float]):
        self._trans_value = trans_value

        space = self.space
        if space == "native":
            self._value = trans_value
        elif space == "log":
            self._value = exp_shift_mirror(
                trans_value, lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )
        elif space == "logit":
            self._value = sgm(
                trans_value, lower_bound=self.lower_bound, upper_bound=self.upper_bound
            )

    @property
    def prior_mean(self):
        return self._prior_mean

    @prior_mean.setter
    def prior_mean(self, prior_mean: Optional[float]):
        self._prior_mean: Optional[float] = prior_mean

        if prior_mean is not None:
            space = self.space
            if space == "native":
                self._trans_prior_mean = prior_mean
            elif space == "log":
                self._trans_prior_mean = log_shift_mirror(
                    prior_mean,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                )
            elif space == "logit":
                self._trans_prior_mean = logit(
                    prior_mean,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                )
        else:
            self._trans_prior_mean = None  # type:ignore
            self._trans_prior_precision: Optional[float] = None

    @property
    def trans_prior_mean(self):
        return self._trans_prior_mean

    @trans_prior_mean.setter
    def trans_prior_mean(self, trans_prior_mean: Optional[float]):
        self._trans_prior_mean = trans_prior_mean  # type: ignore

        if trans_prior_mean is not None:
            space = self.space
            if space == "native":
                self._prior_mean = trans_prior_mean
            elif space == "log":
                self._prior_mean = exp_shift_mirror(
                    trans_prior_mean,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                )
            elif space == "logit":
                self._prior_mean = sgm(
                    trans_prior_mean,
                    lower_bound=self.lower_bound,
                    upper_bound=self.upper_bound,
                )
        else:
            self._prior_mean = None
            self._trans_prior_precision = None

    @property
    def trans_prior_precision(self):
        return self._trans_prior_precision

    @trans_prior_precision.setter
    def trans_prior_precision(self, trans_prior_precision: Optional[float]):
        self._trans_prior_precision = trans_prior_precision

        if trans_prior_precision is None:
            self._prior_mean = None  # type: ignore
            self._trans_prior_mean = None  # type: ignore

    def log_prior(self):
        try:
            return -gaussian_surprise(
                self.trans_value, self.trans_prior_mean, self.trans_prior_precision
            )
        except AttributeError as e:
            raise ModelConfigurationError(
                "trans_prior_mean and trans_prior_precision attributes "
                + "must\nbe specified for method log_prior to return "
                + "a value."
            ) from e


def exp_shift_mirror(x, *, lower_bound: float = 0, upper_bound: Optional[float] = None):
    """The (shifted and mirrored) exponential function"""
    if upper_bound is not None:
        return -exp(x) + upper_bound
    else:
        return exp(x) + lower_bound


def log_shift_mirror(x, *, lower_bound: float = 0, upper_bound: Optional[float] = None):
    """The (shifted and mirrored) natural logarithm"""
    if upper_bound is not None:
        if x > upper_bound:
            raise LogArgumentError(
                "Log argument may not be greater than `upper_bound`."
            )
        elif x == upper_bound:
            return -np.inf
        else:
            return log(-x + upper_bound)
    else:
        if x < lower_bound:
            raise LogArgumentError("Log argument may not be less than `lower_bound`.")
        elif x == lower_bound:
            return -np.inf
        else:
            return log(x - lower_bound)


def sgm(x, *, lower_bound: float = 0, upper_bound: float = 1):
    """The logistic sigmoid function"""
    return (upper_bound - lower_bound) / (1 + exp(-x)) + lower_bound


def logit(x, *, lower_bound: float = 0, upper_bound: float = 1):
    """The logistic function"""
    if x < lower_bound:
        raise LogitArgumentError("Logit argmument may not be less than `lower_bound`.")
    elif x > upper_bound:
        raise LogitArgumentError(
            "Logit argmument may not be greater than `upper_bound`."
        )
    elif x == lower_bound:
        return -np.inf
    elif x == upper_bound:
        return np.inf
    else:
        return log((x - lower_bound) / (upper_bound - x))


def gaussian(x: float, mu: float, pi: float):
    """The Gaussian density as defined by mean and precision"""
    return pi / sqrt(2 * np.pi) * exp(-pi / 2 * (x - mu) ** 2)


def gaussian_surprise(x: float, muhat: float, pihat: float):
    """Surprise at an outcome under a Gaussian prediction"""
    return 0.5 * (log(2 * np.pi) - log(pihat) + pihat * (x - muhat) ** 2)


def binary_surprise(x: float, muhat: float):
    """Surprise at a binary outcome"""
    if x == 1:
        return -log(1 - muhat)
    if x == 0:
        return -log(muhat)
    else:
        raise OutcomeValueError("Outcome needs to be either 0 or 1.")


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
