<img src="docs/source/images/logo.png" align="center" alt="hgf" VSPACE=30>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/ilabcode/ghgf/blob/master/LICENSE) [![codecov](https://codecov.io/gh/ilabcode/ghgf/branch/master/graph/badge.svg)](https://codecov.io/gh/ilabcode/ghgf) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# The multilevel, generalized and nodalized Hierarchical Gaussian Filter for predictive coding

gHGF is a library for generalized and nodalized Hierarchical Gaussian Filters for predictive coding written in [JAX](https://jax.readthedocs.io/en/latest/jax.html). The library consists in a set of function to create graphs of interconnected nodes and recursively update them using prediction errors under new observations. The codebase is also compatible with other JAX libraries to perform Hamiltonian Monte Carlo ([PyMC](https://www.pymc.io/welcome.html), [Blackjax](https://blackjax-devs.github.io/blackjax/)) or gradient descent for parameters estimation.

📖 [Documentation](https://ilabcode.github.io/ghgf/)  
🎓 [Theory](https://ilabcode.github.io/ghgf/theory.html)  
✏️ [Examples](https://ilabcode.github.io/ghgf/tutorials.html)  

## Getting started

### Installation

The latest version of **ghgf** can be installed from the GitHub folder:

`pip install “git+https://github.com/ilabcode/ghgf.git”`

### How does it works?

A Hierarchical Gaussian Filter is a hierarchy of interdependent nodes. Each node can (optionnaly) inherite its values and/or variability for other node higer in the hierarchy. The observation of an input data in the entry node at the lower part of the hierarchy trigger a recursuve update of the nodes values.

A node is formally defined as a Python tuple containing 3 variables:

1. A `parameter` dictionary containing the node parameters (value, precision and parameters controlling the dependencies from values and variability parents).
2. A value parent (optional).
3. A volatility parent (optional).

![Figure1](./docs/source/images/genmod.png)

Value parent (`vapa`) and volatility parent (`vopa`) are also nodes that can have value and/or volatility parents.

The node structure consists of nodes embedding other nodes hierarchically (i.e. here tuples containing other tuples). A generalization of the "standard" Hierarchical Gaussian Filter is any hierarchical structure containing an arbitrary number of nodes, inputs and (linear or non-linear) transformation between nodes. The structure ends when a an orphean node is declared (a node that has no value and no volatility parents).

### Example

Fitting a continuous 3 levels HGF model on a time series.

```python
from numpy import loadtxt
from ghgf.model import HGF
from ghgf import load_data
import jax.numpy as jnp

# Load time series example data
timeserie = load_data("continuous")

# Format input data and add a time vector 
data = jnp.array(
    [
        timeserie, 
        jnp.arange(1, len(timeserie) + 1, dtype=float)
        ]
    ).T

# This is where we define all the model parameters - You can control the value of
# different variables at different levels using the corresponding dictionary.
hgf_model = HGF(
    n_levels=3,
    initial_mu={"1": 1.04, "2": 1.0, "3": 1.0},
    initial_pi={"1": 1e4, "2": 1e1, "3": 1.0},
    omega={"1": -13.0, "2": -2.0, "3": -2.0},
    rho={"1": 0.0, "2": 0.0},
    kappas={"1": 1.0, "2": 1.0},
)

```

`
Fitting the continuous Hierarchical Gaussian Filter (JAX) with 2 levels.
`

```python
jaxhgf.input_data(input_data=data)
```

Get the surprise associated with this model.

```python
jaxhgf.surprise()
```

`
DeviceArray(-1922.2267, dtype=float32)
`

Plot the beliefs trajectories.

```python
jaxhgf.plot_trajectories()
```

![png](./docs/source/images/trajectories.png)

# Acknoledgements

This implementation of the Hierarchical Gaussian Filter was largely inspired by the original [Matlab version](https://translationalneuromodeling.github.io/tapas). A Julia implementation is also available [here](https://github.com/ilabcode/HGF.jl).

## References

1. Mathys, C. (2011). A Bayesian foundation for individual learning under uncertainty. In Frontiers in Human Neuroscience (Vol. 5). Frontiers Media SA. https://doi.org/10.3389/fnhum.2011.00039
2. Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the hierarchical Gaussian filter. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00825
3. Powers, A. R., Mathys, C., & Corlett, P. R. (2017). Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors. Science (New York, N.Y.), 357(6351), 596–600. https://doi.org/10.1126/science.aan3458
