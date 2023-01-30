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
# Multilevel embeding of Hierarchical Gaussian Filters

```{code-cell} ipython3
from numpy import loadtxt
import numpy as np
from ghgf.distribution import hgf_logp, HGFDistribution
from ghgf import load_data
from ghgf.response import binary_surprise
import jax.numpy as jnp
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(123)
```

In the previous tutorials, we interoduced the binary and continuous Hierarchical Gaussian Filters (HGF) with 2 or 3 levels of volatility. Those models are operating at the agent level (i.e. the observations and the parameters being estimated are the observations and the parameters of one agent operating in the environment). However, mode concret use cases of the HGF could require to make inference at the group-level (e.g. comparing the hyper-parameters of different group of participants undergoing the same task). Such example are standard practice in Bayesian cognitive modelling {cite:p}`2014:lee` and require to embede the fitting of HGF model in a Bayesian networks. This is partially what we have been doing in the previous tutorials when we estimated the model's parameters using MCMC sampling. Here, we are going to extend this principle and and fit many models (i.e. many participant) at a time and estimate both parameters and hyper-parameters posterior densities.

+++

```{note} Automatic broadcasting of model parameters
Estimating group-level parameters in the context of a graphical probabilistic model require to fit multiple models at the same time, either on different input data, or on the same data with different parameters, or on different datasets with different parameters. This steps is handled natively both by the `:py:class:ghgf.distribution.hgf_logp` class and the `:py:class:ghgf.distribution.HGFDistribution` class through an automated [broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) approach. When a list of *n* input time series is provided, the function will automatically apply *n* models using the provided parameters. If for some parameters an array of length *n* is provided, each model will use the n-th value as parameter.
```

+++

## Continuous HGF
### Simulate a dataset

```{code-cell} ipython3
n_data = 6
dataset = []
for participant in range(n_data):
    input_data = []
    kappa_1 = 1.0
    omega_1 = -10.0
    omega_2 = -10.0
    mu_1 = 0.0
    mu_2 = 0.0
    pi_1 = 1e4
    pi_2 = 1e1

    for i in range(100):
        
        # x2
        pi_2 = np.exp(omega_2)
        mu_2 = np.random.normal(mu_2, pi_2**.5)

        # x1
        pi_1 = np.exp(kappa_1 * mu_2 + omega_1)
        mu_1 = np.random.normal(mu_1, pi_1**.5)
        
        # input node
        u = np.random.normal(mu_1, 1e-4**.5)
        input_data.append(u)

    dataset.append(input_data)
```

```{code-cell} ipython3
for rw in dataset:
    plt.plot(rw)
```

+++ {"tags": []}

## Embedding a serie of HGFs in a graphical model

+++

Here, we are goingin to estimate the group-level value of the `omega_1` parameter. The dataset consist in 3 time series derived from the classic USD-SWF conversion rate example. Every time series will be fitted to an HGF model where the `omega_1` parameter has to be estimated and the other parameters are fixed.

```{code-cell} ipython3
hgf_logp_op = HGFDistribution(
    n_levels=2,
    model_type="continuous",
    input_data=dataset,
)
```

```{code-cell} ipython3
with pm.Model() as model:
    
    # Hypterpriors
    #-------------
    #mu_omega_1 = pm.Normal("mu_omega_1", mu=.0, sigma=1.0)
    #sigma_omega_1 = pm.Uniform("sigma_omega_1", .5, 10)
    
    # Priors
    #-------
    #normal_dist = pm.Normal.dist(mu=-5, sigma=2, shape=n_data)
    #omega_2 = pm.Censored("omega_2", normal_dist, lower=-20.0, upper=0.0, shape=n_data)
    #omega_1 = pm.Normal("omega_1", mu=mu_omega_1, sigma=sigma_omega_1, shape=n_data)
    omega_1 = pm.Normal("omega_1", mu=0.0, sigma=2.0, shape=n_data)
    #omega_1 = pm.Deterministic("omega_1", norm * sigma_omega_1 + mu_omega_1)

    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            omega_1=omega_1,
            omega_2=-10.0,
            omega_input=np.log(1e-4),
            rho_1=0.0,
            rho_2=0.0,
            pi_1=1e4,
            pi_2=1e1,
            mu_1=0.0,
            mu_2=0.0,
            kappa_1=1.0,
            bias=0.0,
            omega_3=jnp.nan,
            rho_3=jnp.nan,
            pi_3=jnp.nan,
            mu_3=jnp.nan,
            kappa_2=jnp.nan
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
az.plot_trace(idata);
plt.tight_layout()
```

```{code-cell} ipython3
az.summary(idata)
```

## Binary HGF

+++

### Simulate a dataset

```{code-cell} ipython3
n_data = 10
dataset = []
for participant in range(n_data):
    input_data = []
    omega_2 = -1.0
    mu_2 = -1.0

    for i in range(1000):
        
        # x2
        pi_2 = np.exp(omega_2)
        mu_2 = np.random.normal(mu_2, pi_2**.5)
        if mu_2%100 == 0:
            mu_2 = 1.0 if mu_2/100%2 else -1.0

        # x1
        s2 = 1/(1+np.exp(-mu_2))  # sigmoid function
        u = np.random.binomial(n=1, p=s2)       
        input_data.append(u)

    dataset.append(input_data)
```

+++ {"tags": []}

### Without hyper-priors

```{code-cell} ipython3
hgf_logp_op = HGFDistribution(
    n_levels=2,
    model_type="binary",
    input_data=dataset,
    response_function=binary_surprise,
)
```

```{code-cell} ipython3
with pm.Model() as two_levels_binary_hgf:

    omega_2 = pm.Uniform("omega_2", -5.0, 0.0, shape=n_data)

    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            omega_1=jnp.nan,
            omega_2=omega_2,
            omega_input=jnp.nan,
            rho_1=0.0,
            rho_2=0.0,
            pi_1=0.0,
            pi_2=1e4,
            mu_1=jnp.nan,
            mu_2=0.0,
            kappa_1=1.0,
            bias=0.0,
            omega_3=jnp.nan,
            rho_3=jnp.nan,
            pi_3=jnp.nan,
            mu_3=jnp.nan,
            kappa_2=jnp.nan
        ),
    )
```

#### Visualizing the model

```{code-cell} ipython3
pm.model_to_graphviz(two_levels_binary_hgf)
```

#### Sampling

```{code-cell} ipython3
with two_levels_binary_hgf:
    two_level_hgf_idata = pm.sample(chains=4)
```

```{code-cell} ipython3
az.plot_trace(two_level_hgf_idata);
plt.tight_layout()
```

```{code-cell} ipython3
az.summary(two_level_hgf_idata)
```

+++ {"tags": []}

### With hyper-priors

```{code-cell} ipython3
hgf_logp_op = HGFDistribution(
    n_levels=2,
    model_type="binary",
    input_data=data,
    response_function=binary_surprise,
)
```

```{code-cell} ipython3
with pm.Model() as two_levels_binary_hgf:

    # hyper-parameters
    mu_omega = pm.Uniform("mu_omega", -3.0, 0.0)
    sigma_omega = pm.Uniform("sigma_omega", .2, 10.0)
    
    # parameters
    omega_2 = pm.Normal("omega_2", mu_omega, sigma_omega, shape=n_data)

    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            omega_1=jnp.nan,
            omega_2=omega_2,
            omega_input=jnp.nan,
            rho_1=0.0,
            rho_2=0.0,
            pi_1=0.0,
            pi_2=1e4,
            mu_1=jnp.nan,
            mu_2=0.5,
            kappa_1=1.0,
            bias=0.0,
            omega_3=jnp.nan,
            rho_3=jnp.nan,
            pi_3=jnp.nan,
            mu_3=jnp.nan,
            kappa_2=jnp.nan
        ),
    )
```

#### Visualizing the model

```{code-cell} ipython3
pm.model_to_graphviz(two_levels_binary_hgf)
```

#### Sampling

```{code-cell} ipython3
with two_levels_binary_hgf:
    two_level_hgf_idata = pm.sample(chains=4)
```

```{code-cell} ipython3
az.plot_trace(two_level_hgf_idata);
plt.tight_layout()
```

```{code-cell} ipython3
az.summary(two_level_hgf_idata)
```

```{code-cell} ipython3

```
