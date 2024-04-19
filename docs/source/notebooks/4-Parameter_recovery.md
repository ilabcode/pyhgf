---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(parameters_recovery)=
# Recovering computational parameters from observed behaviours

+++ {"editable": true, "slideshow": {"slide_type": ""}}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/4-Parameter_recovery.ipynb)

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-cell]
---
%%capture
import sys

if 'google.colab' in sys.modules:
    !pip install pyhgf watermark
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
from numpy import loadtxt

from pyhgf import load_data
from pyhgf.distribution import HGFDistribution, hgf_logp
from pyhgf.model import HGF
from pyhgf.response import binary_softmax_inverse_temperature
```

```{code-cell} ipython3
np.random.seed(123)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

An important application of Hierarchical Gaussian Filters consists in the inference of computational parameters from observed behaviours, as well as the inference of data-generating models (e.g. are the participants answering randomly or are they learning environmental volatilities that are better approached with a Rescorla-Wagner or a Hierarchical Gaussian Filter?). **Parameter recovery** refers to the ability to recover true data-generating parameters; **model recovery** refers to the ability to correctly identify the true data-generating model using model comparison techniques. It is often a good idea to test parameter/model recovery of a computational model using simulated data before applying this model to experimental data {citep}`RobertCollins2019`. In this tutorial, we demonstrate how to recover some parameters of the generative model of the Hierarchical Gaussian Filter.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Simulate behaviours from a one-armed bandit task
Using a given task structure, we simulate behaviours from a group of participants assuming that they are updating beliefs of environmental volatility using a two-level Hierarchical Gaussian Filter, using a simple sigmoid as a response function parametrized by an inverse temperature parameter. For each participant, the inverse temperature and the tonic volatility at the second level are free parameters that will be estimated during the inference step.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
u, _ = load_data("binary")  # the vector encoding the presence/absence of association

N = 20  # the number of agents to simulate

# sample one value for the inverse temperature (here in log space) and simulate responses
temperatures = np.linspace(0.5, 6.0, num=N)

# sample one new value of the tonic volatility at the second level and fit to observations
volatilities = np.linspace(-6.0, -1.0, num=N)
```

```{code-cell} ipython3
def sigmoid(x, temperature):
    """The sigmoid response function with an inverse temperature parameter."""
    return (x**temperature) / (x**temperature + (1-x)**temperature)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
# create just one default network - we will simply change the values of interest before fitting to save time
agent = HGF(
    n_levels=2,
    verbose=False,
    model_type="binary",
    initial_mean={"1": 0.5, "2": 0.0},
)
```

```{code-cell} ipython3
# observations (always the same), simulated decisions, sample values for temperature and volatility
responses = []
for i in range(N):
    # set the tonic volatility for this agent and run the perceptual model forward
    agent.attributes[2]["tonic_volatility"] = volatilities[i]
    agent.input_data(input_data=u)

    # get decision probabilities using the belief trajectories
    # and the sigmoid decision function with inverse temperature
    p = sigmoid(
        x=agent.node_trajectories[1]["expected_mean"], temperature=temperatures[i]
    )

    # save the observations and decisions separately
    responses.append(np.random.binomial(p=p, n=1))
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Inference from the simulated behaviours

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
hgf_logp_op = HGFDistribution(
    n_levels=2,
    model_type="binary",
    input_data=[u] * N,  # the inputs are the same for all agents - just duplicate the array
    response_function=binary_softmax_inverse_temperature,
    response_function_inputs=responses,
)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Here, we are not assuming hyperpriors to ensure that individual estimates are independent and avoid hierarchical partial pooling.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with pm.Model() as two_levels_binary_hgf:

    # tonic volatility
    volatility = pm.Normal.dist(-3.0, 5, shape=N)
    censored_volatility = pm.Censored("censored_volatility", volatility, lower=-8, upper=2)

    # inverse temperature
    inverse_temperature = pm.Uniform("inverse_temperature", .1, 80, shape=N, initval=np.ones(N))

    # The multi-HGF distribution
    # --------------------------
    pm.Potential(
        "hgf_loglike",
        hgf_logp_op(
            tonic_volatility_2=censored_volatility,
            response_function_parameters=inverse_temperature,
        ),
    )
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
with two_levels_binary_hgf:
    two_level_hgf_idata = pm.sample(chains=2, cores=1)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Visualizing parameters recovery

+++ {"editable": true, "slideshow": {"slide_type": ""}}

A successful parameter recovery is usually inferred from the scatterplot of simulated values and inferred values of the parameters. Here, we can see that the model can recover fairly accurate values close to the underlying parameters. Additionally, we can report the coefficient of correlation between the two variables, as a more objective measure of correspondence.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
fig, axs = plt.subplots(figsize=(12, 5), ncols=2)

axs[0].plot([-6.0, 0.0], [-6.0, 0.0], color="grey", linestyle="--", zorder=-1)
axs[1].plot([0.0, 7.0], [0.0, 7.0], color="grey", linestyle="--", zorder=-1)

for var_name, refs, idx in zip(
    ["censored_volatility", "inverse_temperature"], 
    [volatilities, temperatures], 
    [0, 1], 
):
    inferred_parameters = az.summary(two_level_hgf_idata, var_names=var_name)["mean"].tolist()

    sns.kdeplot(
        x=refs,
        y=inferred_parameters,
        ax=axs[idx],
        fill=True,
        cmap="Reds" if var_name == "censored_volatility" else "Greens",
        alpha=.4
    )

    axs[idx].scatter(
        refs,
        az.summary(two_level_hgf_idata, var_names=var_name)["mean"].tolist(),
        s=60,
        alpha=.8,
        edgecolors="k",
        color= "#c44e52" if var_name == "censored_volatility" else "#55a868"
    )

    axs[idx].grid(True, linestyle='--')
    axs[idx].set_xlabel("Simulated parameter")
    axs[idx].set_ylabel("Infered parameter")
    

axs[0].set_title("Second level tonic volatility")
axs[1].set_title("Inverse temperature")
sns.despine()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# System configuration

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
%load_ext watermark
%watermark -n -u -v -iv -w -p pyhgf,jax,jaxlib
```
