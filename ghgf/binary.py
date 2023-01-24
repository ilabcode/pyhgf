# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit
from jax.interpreters.xla import DeviceArray
from jax.lax import cond

from ghgf.continuous import continuous_node_update
from ghgf.typing import NodeType, ParametersType, ParentsType


def binary_node_update(
    node_parameters: ParametersType,
    value_parents: ParentsType,
    volatility_parents: ParentsType,
    old_time: Union[float, DeviceArray],
    new_time: Union[float, DeviceArray],
) -> NodeType:
    """Update the value and volatility parents of a binary node.
    
    If the parents have value and/or volatility parents, they will be updated
    recursively.

    Updating the node's parents is a two step process:
        1. Update value parent(s) and their parents (if provided).
        2. Update volatility parent(s) and their parents (if provided).

    Then returns the new node tuple `(parameters, value_parents, volatility_parents)`.

    Parameters
    ----------
    node_parameters : dict
        The node parameters is a dictionary containing the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.

        `"psis"` is a tuple with same length than value parents.
        `"kappas"` is a tuple with same length than volatility parents.
    value_parents : tuple | None
        The value parent(s) (optional).
    volatility_parents : tuple | None
        The volatility parent(s) (optional).
    old_time : DeviceArray
        The previous time stamp.
    new_time : DeviceArray
        The previous time stamp.

    Returns
    -------
    node_parameters : dict
        The new node parameters. It contains the following keys:
        `"pihat", "pi", "muhat", "mu", "nu", "psis", "omega"`.
    value_parents : tuple | None
        The new value parent (optional).
    volatility_parents : tuple | None
        The new volatility parent (optional).
    old_time : DeviceArray
        The previous time stamp.
    new_time : DeviceArray
        The previous time stamp.

    """
    # Return here if no parents node are provided
    if (value_parents is None) and (volatility_parents is None):
        return node_parameters, value_parents, volatility_parents

    # Time interval
    t = jnp.subtract(new_time, old_time)

    pihat: DeviceArray = node_parameters["pihat"]
    vape = jnp.subtract(node_parameters["mu"], node_parameters["muhat"])

    #######################################
    # Update the continuous value parents #
    #######################################
    if value_parents is not None:

        # Unpack this node's (x2) parameters, value and volatility parents
        va_pa_node_parameters: ParametersType
        va_pa_value_parents: Tuple[NodeType]
        va_pa_volatility_parents: Tuple[NodeType]
        (
            va_pa_node_parameters,
            va_pa_value_parents,
            va_pa_volatility_parents,
        ) = value_parents[0]

        # 1. get pihat_pa and nu_pa from the value parent (x2)

        # 1.1 get new_nu (x2)
        # 1.1.1 get logvol
        logvol: DeviceArray = va_pa_node_parameters["omega"]

        # 1.1.2 Look at the (optional) va_pa's volatility parents
        # and update logvol accordingly
        if va_pa_volatility_parents is not None:
            for va_pa_vo_pa, k in zip(
                va_pa_volatility_parents, va_pa_node_parameters["kappas"]
            ):
                logvol += k * va_pa_vo_pa[0]["mu"]

        # 1.1.3 Compute new_nu
        nu = t * jnp.exp(logvol)
        new_nu = jnp.where(nu > 1e-128, nu, jnp.nan)

        # 1.2 Compute new value for nu and pihat
        pihat_pa, nu_pa = [1 / (1 / va_pa_node_parameters["pi"] + new_nu), new_nu]

        # 2.
        pi_pa = pihat_pa + 1 / pihat

        # 3. get muhat_pa from value parent (x2)

        # 3.1
        driftrate = va_pa_node_parameters["rho"]

        # 3.2 Look at the (optional) va_pa's value parents
        # and update driftrate accordingly
        if va_pa_value_parents is not None:
            psi: DeviceArray
            for va_pa_va_pa, psi in zip(
                va_pa_value_parents, va_pa_node_parameters["psis"]
            ):
                _mu: DeviceArray = va_pa_va_pa[0]["mu"]
                driftrate += psi * _mu
        # 3.3
        muhat_pa = va_pa_node_parameters["mu"] + t * driftrate

        # 4.
        mu_pa = muhat_pa + vape / pi_pa  # This line differs from the continuous input

        # 5. Update node's parameters and node's parents recursively
        va_pa_node_parameters["pihat"] = pihat_pa
        va_pa_node_parameters["pi"] = pi_pa
        va_pa_node_parameters["muhat"] = muhat_pa
        va_pa_node_parameters["mu"] = mu_pa
        va_pa_node_parameters["nu"] = nu_pa

        (
            new_va_pa_node_parameters,
            new_va_pa_value_parents,
            new_va_pa_volatility_parents,
        ) = continuous_node_update(
            node_parameters=va_pa_node_parameters,
            value_parents=va_pa_value_parents,
            volatility_parents=va_pa_volatility_parents,
            old_time=old_time,
            new_time=new_time,
        )

        new_value_parents = ()
        new_value_parents += (
            (
                new_va_pa_node_parameters,
                new_va_pa_value_parents,
                new_va_pa_volatility_parents,
            ),
        )

    else:
        new_value_parents = value_parents

    # A binary node only have value parents
    new_volatility_parents = None

    ##########################
    # Update node parameters #
    ##########################
    new_node_parameters = node_parameters

    return new_node_parameters, new_value_parents, new_volatility_parents


def binary_input_update(
    input_node: NodeType,
    value: DeviceArray,
    new_time: DeviceArray,
    old_time: DeviceArray,
) -> Optional[Tuple[DeviceArray, NodeType]]:
    """Update the input node structure given one observation.

    This function is the entry level of the model fitting. It update the partents of
    the input node and then call py:func:`ghgf.binary.binary_node_update` to update the
    rest of the node structure.

    Parameters
    ----------
    input_node : NodeType
        The parameter, value parent and volatility parent of the input node. The
        volatility and value parents contain their own value and volatility parents,
        this structure being nested up to the higher level of the hierarchy.
    value : DeviceArray
        The new input value that is observed by the model at time t=`new_time`.
    new_time : DeviceArray
        The current time point.
    old_time : DeviceArray
        The time point of the previous observed value.

    Returns
    -------
    surprise : jnp.DeviceArray
        The gaussian surprise given the value(s) presented at t=`new_time`.
    new_input_node : tuple
        The input node structure after recursively updating all the nodes.

    See Also
    --------
    update_continuous_parents, update_continuous_input_parents

    """
    input_node_parameters: ParametersType
    value_parents: ParentsType
    volatility_parents: ParentsType
    input_node_parameters, value_parents, volatility_parents = input_node

    if (value_parents is None) and (volatility_parents is None):
        return None

    # Time interval
    t = jnp.subtract(new_time, old_time)

    pihat: DeviceArray = input_node_parameters["pihat"]

    ################################
    # Update parents (binary node) #
    ################################

    if value_parents is not None:

        # Unpack this binary node's parameters, value and volatility parents (None)
        va_pa_node_parameters: ParametersType
        va_pa_value_parents: ParentsType
        va_pa_volatility_parents: ParentsType
        (
            va_pa_node_parameters,
            va_pa_value_parents,
            va_pa_volatility_parents,
        ) = value_parents[0]

        # 1. Compute new muhat_pa and pihat_pa from binary node parent
        # ------------------------------------------------------------

        # 1.1 Compute new_muhat from continuous node parent (x2)

        # 1.1.1 get rho from the value parent of the binary node (x2)
        driftrate: DeviceArray = va_pa_value_parents[0][0]["rho"]

        # 1.1.2 Look at the (optional) va_pa's value parents (x3)
        # and update the driftrate accordingly
        if va_pa_value_parents[0][1] is not None:
            for va_pa_va_pa in va_pa_value_parents[0][1]:
                # For each x2's value parents (optional)
                _psi: DeviceArray = va_pa_value_parents[0][0]["psis"]
                _mu: DeviceArray = va_pa_va_pa[0]["mu"]
                driftrate += _psi * _mu

        # 1.1.3 compute new_muhat
        muhat_va_pa = va_pa_value_parents[0][0]["mu"] + t * driftrate

        muhat_va_pa = sgm(muhat_va_pa)
        pihat_va_pa = 1 / (muhat_va_pa * (1 - muhat_va_pa))

        # 2. Compute surprise
        # -------------------
        eta0: DeviceArray = input_node_parameters["eta0"]
        eta1: DeviceArray = input_node_parameters["eta1"]

        mu_va_pa, pi_va_pa, surprise = cond(
            pihat == jnp.inf,
            input_surprise_inf,
            input_surprise_reg,
            (pihat, value, eta1, eta0, muhat_va_pa),
        )

        # Update value parent's parameters
        va_pa_node_parameters["pihat"] = pihat_va_pa
        va_pa_node_parameters["pi"] = pi_va_pa
        va_pa_node_parameters["muhat"] = muhat_va_pa
        va_pa_node_parameters["mu"] = mu_va_pa

        # 4. Update the binary node parent
        (
            new_va_pa_node_parameters,
            new_va_pa_value_parents,
            new_va_pa_volatility_parents,
        ) = binary_node_update(
            node_parameters=va_pa_node_parameters,
            value_parents=va_pa_value_parents,
            volatility_parents=va_pa_volatility_parents,
            old_time=old_time,
            new_time=new_time,
        )

        new_value_parents: ParentsType = (
            new_va_pa_node_parameters,
            new_va_pa_value_parents,
            new_va_pa_volatility_parents,
        )

    else:
        new_value_parents = value_parents  # type:ignore

    input_node = input_node_parameters, (new_value_parents,), None

    return surprise, input_node


def gaussian_density(
    x: DeviceArray, mu: DeviceArray, pi: DeviceArray
) -> jnp.DeviceArray:
    """Gaussian density as defined by mean and precision."""
    return (
        pi
        / jnp.sqrt(2 * jnp.pi)
        * jnp.exp(jnp.subtract(0, pi) / 2 * (jnp.subtract(x, mu)) ** 2)
    )


def sgm(
    x,
    lower_bound: DeviceArray = jnp.array(0.0),
    upper_bound: DeviceArray = jnp.array(1.0),
):
    """Logistic sigmoid function."""
    return jnp.subtract(upper_bound, lower_bound) / (1 + jnp.exp(-x)) + lower_bound


def binary_surprise(x: DeviceArray, muhat: DeviceArray):
    """Surprise at a binary outcome.

    Parameters
    ----------
    x : jnp.DeviceArray
        The outcome.
    muhat : jnp.DeviceArray
        The mean of the Bernouilli distribution.

    Returns
    -------
    surprise : jnp.DeviceArray
        The surprise.

    """
    return jnp.where(x, -jnp.log(muhat), -jnp.log(jnp.array(1.0) - muhat))


@jit
def loop_binary_inputs(
    res: Tuple[NodeType, Dict[str, DeviceArray]], el: Tuple[DeviceArray, DeviceArray]
) -> Tuple[
    Tuple[NodeType, Dict[str, DeviceArray]], Tuple[NodeType, Dict[str, DeviceArray]]
]:
    """Add new observation and update the node structures.
    
    The HGF function to be scanned by JAX. One time step updating node structure and
    returning the new node structure with time, value and surprise.

    Parameters
    ----------
    res : tuple
        Result from the previous iteration. The node structure and a dictionary results
        from the previous iteration as a tuple in the form:
        `(node_structure, {"time": time, "value": value, "surprise": surprise})`.
    el : tuple
        The current array element. Scan will iterate over a n x 2 DeviceArray with time
        and values (i.e. the input time series).

    """
    # Extract the current iteration variables values and time
    value, new_time = el

    # Extract model previous structure and results {time, value, surprise}
    node_structure, results = res

    # Fit the model with the current time and value variables, given model structure
    surprise, new_node_structure = binary_input_update(  # type:ignore
        input_node=node_structure,
        value=value,
        old_time=results["time"],
        new_time=new_time,
    )

    # the model structure is packed with current time point
    # surprise is accumulated separately
    res = new_node_structure, {"time": new_time, "value": value, "surprise": surprise}

    return res, res  # ("carryover", "accumulated")


def input_surprise_inf(op):
    """Apply special case if pihat is `jnp.inf` (just pass the value through)."""
    (pihat, value, eta1, eta0, muhat_va_pa) = op
    mu_va_pa = value
    pi_va_pa = jnp.inf
    surprise = binary_surprise(value, muhat_va_pa)

    return mu_va_pa, pi_va_pa, surprise


def input_surprise_reg(op):
    """Compute the surprise, mu_va_pa and pi_va_pa if pihat is not `jnp.inf`."""
    (pihat, value, eta1, eta0, muhat_va_pa) = op

    # Likelihood under eta1
    und1 = jnp.exp(jnp.subtract(0, pihat) / 2 * (jnp.subtract(value, eta1)) ** 2)

    # Likelihood under eta0
    und0 = jnp.exp(jnp.subtract(0, pihat) / 2 * (jnp.subtract(value, eta0)) ** 2)

    # Eq. 39 in Mathys et al. (2014) (i.e., Bayes)
    mu_va_pa = muhat_va_pa * und1 / (muhat_va_pa * und1 + (1 - muhat_va_pa) * und0)
    pi_va_pa = 1 / (mu_va_pa * (1 - mu_va_pa))

    # Surprise
    surprise = -jnp.log(
        muhat_va_pa * gaussian_density(value, eta1, pihat)
        + (1 - muhat_va_pa) * gaussian_density(value, eta0, pihat)
    )

    return mu_va_pa, pi_va_pa, surprise
