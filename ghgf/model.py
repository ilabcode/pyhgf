# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import jax.numpy as jnp
from typing import Optional, Tuple, Dict
from jax.lax import scan
from ghgf.hgf_jax import (
    update_parents,
    update_input_parents,
    gaussian_surprise,
    loop_inputs,
    node_validation,
)


class HGF(object):
    """Generic HGF model"""

    def __init__(
        self,
        n_levels: int = 2,
        model_type: Optional[str] = None,
        initial_mu: Dict[str, float] = {"1": 0.0, "2": 0.0},
        initial_pi: Dict[str, float] = {"1": 1.0, "2": 1.0},
        omega_input: float = jnp.log(1e-4),
        omega: Dict[str, float] = {"1": -10.0, "2": -10.0},
        kappa: Dict[str, float] = {"1": 1.0},
        rho: Optional[Dict[str, float]] = None,
        phi: Optional[Dict[str, float]] = None,
        m: Dict[str, float] = None,
        verbose: bool = True,
    ):

        """The standard n-level HGF for continuous inputs.

        The standard continuous HGF can implements the Gaussian Random Walk and AR1
        perceptual models.

        Parameters
        ----------
        n_levels : int
            The number of hierarchies in the perceptual model (default sets to `2`).
        model_type : str or None
            The model type to use (can be "GRW" or "AR1"). If `model_type` is not provided,
            it is infered from the parameters provided. If both `phi` and `rho` are None
            or dictionnary, an error is returned.
        initial_mu : dict
            Dictionnary containing the initial values for the `initial_mu` parameter at
            different levels of the hierarchy. Defaults set to `{"1": 0.0, "2": 0.0}`
            for a 2-levels model.
        initial_pi : dict
            Dictionnary containing the initial values for the `initial_pi` parameter at
            different levels of the hierarchy. Pis values encode the precision of the
            values at each level (Var = 1/pi) Defaults set to `{"1": 1.0, "2": 1.0}` for
            a 2-levels model.
        omega : dict
            Dictionnary containing the initial values for the `omega` parameter at
            different levels of the hierarchy. Omegas represent the tonic part of the
            variance (the part that is not affected by the parent node). Defaults set to
            `{"1": -10.0, "2": -10.0}` for a 2-levels model. This parameters only when
            `model_type="GRW"`.
        omega_input : float
            Default value sets to `np.log(1e-4)`. Represent the noise associated with the
            input.
        rho : dict | None
            Dictionnary containing the initial values for the `rho` parameter at
            different levels of the hierarchy. Rho represents the drift of the random walk.
            Only required when `model_type="GRW"`. Defaults set all entries to `0`
            according to the number of required levels.
        kappa : dict
            Dictionnary containing the initial values for the `kappa` parameter at
            different levels of the hierarchy. Kappa represents the phasic part of the
            variance (the part that is affected by the parents nodes) and will defines the
            strenght of the connection between the node and the parent node. Often fixed
            to 1. Defaults set to `{"1": 1.0}` for a 2-levels model. Only required when
            `model_type="GRW"`.
        phi : dict | None
            Dictionnary containing the initial values for the `phi` parameter at
            different levels of the hierarchy. Phi should always be between 0 and 1.
            Defaults set all entries to `0` according to the number of required levels.
            `phi` is only required when `model_type="AR1"`.
        m : dict or None
            Dictionnary containing the initial values for the `m` parameter at
            different levels of the hierarchy. Defaults set all entries to `0` according to
            the number of required levels. `m` is only required when `model_type="AR1"`.
        verbose : bool
            Default is `True` (show bar progress).

        Attributes
        ----------
        n_levels : int
            The number of hierarchies in the model. Cannot be less than 2.
        model_type : str
            The model implemented (can be `"AR1"` or `"GRW"`).

        Notes
        -----
        The model used by the perceptual model is defined by the `model_type` parameter
        (can be `"GRW"` or `"AR1"`). If `model_type` is not provided, the class will try to
        determine it automatically by looking at the `rho` and `phi` parameters. If `rho`
        is provided `model_type="GRW"`, if `phi` is provided `model_type="AR1"`. If both
        `phi` and `rho` are `None` an error will be returned.

        Examples
        --------

        """


    def add_nodes(self, nodes: Tuple):
        """Add a custom node structure.

        Parameters
        ----------
        nodes : tuple
            The input node embeding the node hierarchy that will be updated during
            model fit.

        """
        node_validation(nodes, input_node=True)
        self.nodes = nodes

    def input_data(self, input_data):

        # Initialise the first values
        res_init = (
            self.nodes,
            {
                "time": jnp.array(0.0),
                "value": jnp.array(0.0),
                "surprise": jnp.array(0.0),
            },
        )

        # This is where the HGF functions are used to scan the input time series
        last, final = scan(loop_inputs, res_init, input_data)

        # Store ouptut values
        self.last = last  # The last tuple returned
        self.final = final  # The commulative update of the nodes and results


    def surprise(self):

        _, results = self.final
        surprise = jnp.sum(results["surprise"])
        return jnp.where(jnp.isnan(surprise), -jnp.inf, surprise)

    def log_prior(self):
        pass
        # # The log_prior of the parameters
        # - gaussian_surprise(
        #     x=trans_value,
        #     muhat=trans_prior_mean,
        #     pihat=trans_prior_precision
        #     )

        # if self.var_params:
        #     log_prior = 0
        #     for var_param in self.var_params:
        #         log_prior += var_param.log_prior()
        #     return log_prior
        # else:
        #     raise ModelConfigurationError("No variable (i.e., non-fixed) parameters.")

    def log_joint(self):
        return -self.surprise() + self.log_prior()

    # def neg_log_joint_function(self):
    #     def f(trans_values):
    #         trans_values_backup = self.var_param_trans_values
    #         self.var_param_trans_values = trans_values
    #         self.recalculate()

    #         # Check for NaNs in the parameters times series (invalid parameter space)
    #         has_nan = np.array(
    #             [
    #                 np.isnan(
    #                     np.array(
    #                         [
    #                             getattr(self, f"x{i+1}").mus[1:],
    #                             getattr(self, f"x{i+1}").pis[1:],
    #                         ]
    #                     )
    #                 )
    #                 for i in range(self.n_levels)
    #             ]
    #         ).any()

    #         return np.where(has_nan, np.nan, -self.log_joint())

    #     return f
