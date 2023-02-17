<img src="docs/source/images/logo.svg" align="center" alt="hgf" VSPACE=30>

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit) [![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/ilabcode/pyhgf/blob/master/LICENSE) [![codecov](https://codecov.io/gh/ilabcode/pyhgf/branch/master/graph/badge.svg)](https://codecov.io/gh/ilabcode/pyhgf) [![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/) [![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

# The multilevel, generalized and nodalized Hierarchical Gaussian Filter for predictive coding

pyhgf is a Python library for generalized and nodalized Hierarchical Gaussian Filters for predictive coding written on top of [JAX](https://jax.readthedocs.io/en/latest/jax.html). The library consists in a set of function to create graphs of interconnected nodes and recursively update them from precision-weighted prediction errors under new observations. The codebase interface natively with other libraries in the JAX ecosystem or  probabilistic programming tools for Bayesian inference.

* 🎓 [What is a Hierarchical Gaussian Filter?](https://ilabcode.github.io/pyhgf/theory.html)  
* 📖 [API Documentation](https://ilabcode.github.io/pyhgf/)  
* ✏️ [Tutorials and examples](https://ilabcode.github.io/pyhgf/tutorials.html)  

## Getting started

### Installation

The last official release can be download from PIP:

`pip install pyhgf`

The current version under development can be installed from the master branch of the GitHub folder:

`pip install “git+https://github.com/ilabcode/pyhgf.git”`

### How does it works?

A Hierarchical Gaussian Filter consists in a network of probabilistic nodes hierarchically structured where each node can inherit its value and volatility sufficient statistics from other parents node. The presentation of a new observation at the lower level of the hierarchy (i.e. the input node) trigger a recursuve update of the nodes belief throught the bottom-up propagation of precision-weigthed prediction error.

A node is formally defined as a Python tuple containing 3 variables:

* A `parameter` dictionary containing the node parameters (value, precision and parameters controlling the dependencies from values and variability parents).
* A tuple of value parent(s) (optional).
* A tuple of volatility parent(s) (optional).

Value parent and volatility parent are nodes themself and are built from the same principles. A filtering structure consists in nodes embedding other nodes hierarchically. In Python, this is implemented as tuples containing other tuples. Nodes can be parametrized by their sufficient statistic and update rules. One node can have multiple value/volatility parents, and influence multiple children nodes separately. The transformations between nodes can be linear or non-linear (thus a *generalization* of the HGF).

### The Hierarchical Gaussian Filter

The Hierarchical Gaussian Filter for binary and continuous inputs as it was described in Mathys et al. (2011, 2014), and later implemented in the Matlab Tapas toolbox (Frässle et al. 2021), can be seen as a special case of this node structure such as:

![Figure2](./docs/source/images/hgf.png)

The pyhgf package includes pre-implemented standard HGF models that can be used together with other neural network libraries of Bayesian inference tools. It is also possible for the user to build custom network structures that would match specific needs.

### Model fitting

Here we demonstrate how to fit a two-level binary Hierarchical Gaussian filter. The input time series are binary outcome from Iglesias et al. (2013).

```python
from pyhgf.model import HGF
from pyhgf import load_data

# Load time series example data
timeserie = load_data("binary")

# This is where we define all the model parameters - You can control the value of
# different variables at different levels using the corresponding dictionary.
hgf = HGF(
    n_levels=2,
    model_type="binary",
    initial_mu={"1": .0, "2": .5},
    initial_pi={"1": .0, "2": 1e4},
    omega={"1": None, "2": -3.0},
    rho={"1": None, "2": 0.0},
    kappas={"1": None},
    eta0=0.0,
    eta1=1.0,
)

# add new observations
hgf.input_data(input_data=timeserie)

# compute the model's surprise (-log(p))
surprise = hgf.surprise()
print(f"Model's surprise = {surprise}")

# visualization of the belief trajectories
hgf.plot_trajectories()
```

`Creating a binary Hierarchical Gaussian Filter with 2 levels.`  
`Add 320 new binary observations.`  
`Model's surprise = 203.29249572753906`

![png](./docs/source/images/trajectories.png)

# Acknoledgements

This implementation of the Hierarchical Gaussian Filter was largely inspired by the original [Matlab version](https://translationalneuromodeling.github.io/tapas). A Julia implementation of the generalised, nodalised and multilevel HGF is also available [here](https://github.com/ilabcode/HGF.jl).

## References

1. Mathys, C. (2011). A Bayesian foundation for individual learning under uncertainty. In Frontiers in Human Neuroscience (Vol. 5). Frontiers Media SA. https://doi.org/10.3389/fnhum.2011.00039
2. Mathys, C. D., Lomakina, E. I., Daunizeau, J., Iglesias, S., Brodersen, K. H., Friston, K. J., & Stephan, K. E. (2014). Uncertainty in perception and the hierarchical Gaussian filter. Frontiers in Human Neuroscience, 8. https://doi.org/10.3389/fnhum.2014.00825
3. Powers, A. R., Mathys, C., & Corlett, P. R. (2017). Pavlovian conditioning-induced hallucinations result from overweighting of perceptual priors. Science (New York, N.Y.), 357(6351), 596–600. https://doi.org/10.1126/science.aan3458
4. Frässle, S., Aponte, E. A., Bollmann, S., Brodersen, K. H., Do, C. T., Harrison, O. K., Harrison, S. J., Heinzle, J., Iglesias, S., Kasper, L., Lomakina, E. I., Mathys, C., Müller-Schrader, M., Pereira, I., Petzschner, F. H., Raman, S., Schöbi, D., Toussaint, B., Weber, L. A., … Stephan, K. E. (2021). TAPAS: An Open-Source Software Package for Translational Neuromodeling and Computational Psychiatry. In Frontiers in Psychiatry (Vol. 12). Frontiers Media SA. https://doi.org/10.3389/fpsyt.2021.680811
