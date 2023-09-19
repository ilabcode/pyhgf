# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

from typing import Union

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def gaussian_density(x: ArrayLike, mu: ArrayLike, pi: ArrayLike) -> ArrayLike:
    """Gaussian density as defined by mean and precision."""
    return pi / jnp.sqrt(2 * jnp.pi) * jnp.exp(-pi / 2 * (x - mu) ** 2)


def sgm(
    x,
    lower_bound: Union[ArrayLike, float] = 0.0,
    upper_bound: Union[ArrayLike, float] = 1.0,
) -> ArrayLike:
    """Logistic sigmoid function."""
    return (upper_bound - lower_bound) / (1 + jnp.exp(-x)) + lower_bound


def binary_surprise(
    x: Union[float, ArrayLike], muhat: Union[float, ArrayLike]
) -> ArrayLike:
    r"""Surprise at a binary outcome.

    The surprise ellicited by a binary observation :math:`x` under the expected
    probability :math:`\hat{\mu}` is given by:

    .. math::

       \begin{cases}
            -\log(\hat{\mu}),& \text{if } x=1\\
            -\log(1 - \hat{\mu}), & \text{if } x=0\\
        \end{cases}

    Parameters
    ----------
    x :
        The outcome.
    muhat :
        The mean of the Bernoulli distribution.

    Returns
    -------
    surprise :
        The binary surprise.


    Examples
    --------
    >>> from pyhgf.binary import binary_surprise
    >>> binary_surprise(x=1.0, muhat=0.7)
    `Array(0.35667497, dtype=float32, weak_type=True)`

    """
    return jnp.where(x, -jnp.log(muhat), -jnp.log(jnp.array(1.0) - muhat))


def gaussian_surprise(
    x: Union[float, ArrayLike],
    muhat: Union[float, ArrayLike],
    pihat: Union[float, ArrayLike],
) -> Array:
    r"""Surprise at an outcome under a Gaussian prediction.

    The surprise elicited by an observation :math:`x` under a Gaussian distribution
    with expected mean :math:`\hat{\mu}` and expected precision :math:`\hat{\pi}` is
    given by:

    .. math::

       \frac{1}{2} \log(2 \pi) - \log(\hat{\pi}) + \hat{\pi}\sqrt{x - \hat{\mu}}

    where :math:`\pi` is the mathematical constant.

    Parameters
    ----------
    x :
        The outcome.
    muhat :
        The expected mean of the Gaussian distribution.
    pihat :
        The expected precision of the Gaussian distribution.

    Returns
    -------
    surprise :
        The Gaussian surprise.

    Examples
    --------
    >>> from pyhgf.continuous import gaussian_surprise
    >>> gaussian_surprise(x=2.0, muhat=0.0, pihat=1.0)
    `Array(2.9189386, dtype=float32, weak_type=True)`

    """
    return jnp.array(0.5) * (
        jnp.log(jnp.array(2.0) * jnp.pi)
        - jnp.log(pihat)
        + pihat * jnp.square(jnp.subtract(x, muhat))
    )
