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
```

## Imports
Import binary responses from {cite:p}`2013:iglesias`.

```{code-cell} ipython3
timeserie = load_data("binary")

# Format the data input accordingly (a value column and a time column)
data = jnp.array([timeserie, jnp.arange(1, len(timeserie) + 1, dtype=float)]).T
```

## Using a two-levels model
### Create the model

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
two_levels_hgf = two_levels_hgf.input_data(data)
```

### Plot trajectories

```{code-cell} ipython3
two_levels_hgf.plot_trajectories()
```

## Using a three-level model
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
three_levels_hgf = three_levels_hgf.input_data(data)
```

### Plot trajectories

```{code-cell} ipython3
three_levels_hgf.plot_trajectories()
```

```{code-cell} ipython3

```
