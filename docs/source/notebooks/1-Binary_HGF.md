---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(binary_hgf)=
# The binary Hierarchical Gaussian Filter

```{code-cell} ipython3
import jax.numpy as jnp
from ghgf.model import HGF
from ghgf import load_data
import seaborn as sns

sns.set_context("talk")
```

In this notebook, we demonstrate how to use the standard 2-levels and 3-level Hierarchical Gaussian Filters (HGF) for binary inputs. This class of models is identical to the continuous counterpart, with the difference that the input node accepts binary data. Such binary responses are widely used in decision-making studies. Here, we will observe how binary HGFs can track switches in responses probability across the task.

## Imports
We import a time series of binary responses from the decision task described in {cite:p}`2013:iglesias`.

```{code-cell} ipython3
timeserie = load_data("binary")
```

## Using a two-levels Hierarchical Gaussian Filter
### Create the model

The node structure corresponding to the 2-levels and 3-levels Hierarchical Gaussian Filters are automatically generated from `model_type` and `n_levels` using the nodes parameters provided in the dictionaries. Here we are not performing any optimization so thoses parameters are fixed to reasonnable values.

```{code-cell} ipython3
two_levels_hgf = HGF(
    n_levels=2,
    model_type="binary",
    initial_mu={"1": .0, "2": .5},
    initial_pi={"1": .0, "2": 1e4},
    omega={"1": None, "2": -6.0},
    rho={"1": None, "2": 0.0},
    kappas={"1": None},
    eta0=0.0,
    eta1=1.0,
    pihat = jnp.inf,
)
```

### Add data

```{code-cell} ipython3
# Provide new observations
two_levels_hgf = two_levels_hgf.input_data(input_data=timeserie)
```

### Plot trajectories

```{code-cell} ipython3
two_levels_hgf.plot_trajectories()
```

We can see that the surprise will increase when the time series exhibit more unexpected behaviors. The degree to which a given observation is expected will deppends on the expeted value and volatility in the input node, that are influenced by the values of higher order nodes. One way to assess model fit is to look at the total gaussian surprise for each observation. This values can be returned using the `surprise` method:

```{code-cell} ipython3
two_levels_hgf.surprise()
```

## Using a three-level Hierarchical Gaussian Filter
### Create the model

```{code-cell} ipython3
three_levels_hgf = HGF(
    n_levels=3,
    model_type="binary",
    initial_mu={"1": .0, "2": .5, "3": 0.},
    initial_pi={"1": .0, "2": 1e4, "3": 1e1},
    omega={"1": None, "2": -6.0, "3": -2.0},
    rho={"1": None, "2": 0.0, "3": 0.0},
    kappas={"1": None, "2": 1.0},
    eta0=0.0,
    eta1=1.0,
    pihat = jnp.inf,
)
```

### Add data

```{code-cell} ipython3
three_levels_hgf = three_levels_hgf.input_data(input_data=timeserie)
```

### Plot trajectories

```{code-cell} ipython3
three_levels_hgf.plot_trajectories()
```

# System configuration

```{code-cell} ipython3
%load_ext watermark
%watermark -n -u -v -iv -w -p ghgf,jax,jaxlib
```

```{code-cell} ipython3

```
