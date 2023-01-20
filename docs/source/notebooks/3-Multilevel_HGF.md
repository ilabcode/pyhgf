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

(multilevel_hgf)=
# Embeding the HGF in a multilevel model

```{code-cell} ipython3
from numpy import loadtxt
from ghgf.distribution import hgf_logp, HGFDistribution
from ghgf import load_data
import jax.numpy as jnp
import numpy as np
import pymc as pm
import arviz as az
```

In this tutorial, we are going to estimate the group-level probability density distribution of HGF parameters.

+++

## Using the automated broadcasting for models creation
Estimating group-level parameters in the context of a graphical probabilistic model require to fit multiple models at the same time, either on different input data, or on the same data with different parameters, or on different datasets with different parameters. This steps is handled natively both by the `:py:class:ghgf.distribution.hgf_logp` class and the `:py:class:ghgf.distribution.HGFDistribution` class through automated [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html).

```{code-cell} ipython3
# Create an example dataset using slice copies of the original currencies exchange rate time series
timeserie = load_data("continuous")

data = [timeserie[i*100:i*100+100] for i in range(6)]
```

```{code-cell} ipython3
hgf_logp(
    omega_1 = -3.0,
    omega_2 = -3.0,
    omega_input = jnp.log(1e-4),
    rho_1 = 0.0,
    rho_2 = 0.0,
    pi_1 = 1e4,
    pi_2 = 1e1,
    mu_1 = 1.0,
    mu_2 = 0.0,
    kappa_1 = 1.0,
    bias = 0.0,
    input_data = data,
    model_type = "continuous",
    n_levels = 2,
    response_function = None
)
```

+++ {"tags": []}

## Embedding a serie of HGFs in a graphical model

+++

Here, we are goingin to estimate the group-level value of the `omega_1` parameter. The dataset consist in 3 time series derived from the classic USD-SWF conversion rate example. Every time series will be fitted to an HGF model where the `omega_1` parameter has to be estimated and the other parameters are fixed.

```{code-cell} ipython3
hgf_logp_op = HGFDistribution(
    n_levels=2,
    input_data=data,
)
```

```{code-cell} ipython3
with pm.Model() as model:
    
    # Hypterpriors
    #-------------
    #mu_omega_2 = pm.Normal("mu_omega_2", -2.0, 5.0)
    #sigma_omega_2 = 2.0 # pm.Uniform("sigma_omega_2", 0, 10.0)
    
    # Priors
    #-------
    normal_dist2 = pm.Normal.dist(mu=-2.0, sigma=2.0, shape=6)
    omega_2 = pm.Censored("omega_2", normal_dist2, lower=-20.0, upper=2)
    mu_1 = pm.Uniform("mu_1", 0, 2, shape=6)

    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            omega_1=-2.0,
            omega_2=omega_2,
            omega_input=np.log(1e-4),
            rho_1=0.0,
            rho_2=0.0,
            pi_1=1e4,
            pi_2=1e1,
            mu_1=mu_1,
            mu_2=0.0,
            kappa_1=1.0,
            bias=0.0,
        ),
    )
```

```{code-cell} ipython3
pm.model_to_graphviz(model)
```

```{code-cell} ipython3
with model:
    idata = pm.sample(chains=4, cores=4, tune=1000)
```

```{code-cell} ipython3
az.plot_trace(idata)
```

```{code-cell} ipython3

```
