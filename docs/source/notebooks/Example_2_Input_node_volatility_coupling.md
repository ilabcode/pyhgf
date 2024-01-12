---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

(example_2)=
# Example 2: Estimating the mean and precision of an input node

+++ {"editable": true, "slideshow": {"slide_type": ""}}

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/Example_2_Input_node_volatility_coupling.ipynb)

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
    !pip install pyhgf
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
from scipy.stats import norm

from pyhgf.distribution import HGFDistribution
from pyhgf.model import HGF
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Where the standard continuous HGF assumes a known precision in the input node (usually set to something high), this assumption can be relaxed and the filter can also try to estimate this quantity from the data. In this notebook, we demonstrate how we can infer the value of the mean, of the precision, or both value at the same time, using the appropriate value and volatility coupling parents.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Unkown mean, known precision

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```{hint}
The {ref}`continuous_hgf` is an example of a model assuming a continuous input with known precision and unknown mean. It is further assumed that the mean is changing overtime, and we want the model to track this rate of change by adding a volatility node on the top of the value parent (two-level continuous HGF), and event track the rate of change of this rate of change by adding another volatility parent (three-level continuous HGF).
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
np.random.seed(123)
dist_mean, dist_std = 5, 1
input_data = np.random.normal(loc=dist_mean, scale=dist_std, size=1000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mean_hgf = (
    HGF(model_type=None, update_type="standard")
    .add_nodes(kind="continuous-input", node_parameters={'input_noise': 1.0, "expected_precision": 1.0})
    .add_nodes(value_children=0, node_parameters={"tonic_volatility": -8.0})
    .init()
).input_data(input_data)
mean_hgf.plot_network()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

```{note}
We are setting the tonic volatility to something low for visualization purposes, but changing this value can make the model learn in fewer iterations.
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# get the nodes trajectories
df = mean_hgf.to_pandas()

fig, ax = plt.subplots(figsize=(12, 5))

x = np.linspace(-10, 10, 1000)
for i, color in zip([0, 2, 5, 10, 50, 500], plt.cm.Greys(np.linspace(.2, 1, 6))):

    # extract the sufficient statistics from the input node (and parents)
    mean = df.x_1_expected_mean.iloc[i]
    std = np.sqrt(
        1/(mean_hgf.attributes[0]["expected_precision"])
    )

    # the model expectations
    ax.plot(x, norm(mean, std).pdf(x), color=color, label=i)

# the sampling distribution
ax.fill_between(x, norm(dist_mean, dist_std).pdf(x), color="#582766", alpha=.2)

ax.legend(title="Iterations")
ax.set_xlabel("Input (u)")
ax.set_ylabel("Density")
plt.grid(linestyle=":")
sns.despine()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Kown mean, unknown precision

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## Unkown mean, unknown precision

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
np.random.seed(123)
dist_mean, dist_std = 5, 1
input_data = np.random.normal(loc=dist_mean, scale=dist_std, size=1000)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
mean_precision_hgf = (
    HGF(model_type=None)
    .add_nodes(kind="continuous-input", node_parameters={'input_noise': 0.01, "expected_precision": 0.01})
    .add_nodes(value_children=0, node_parameters={"tonic_volatility": -6.0})
    .add_nodes(volatility_children=0, node_parameters={"tonic_volatility": -6.0})
    .init()
).input_data(input_data)
mean_precision_hgf.plot_network()
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
tags: [hide-input]
---
# get the nodes trajectories
df = mean_precision_hgf.to_pandas()

fig, ax = plt.subplots(figsize=(12, 5))

x = np.linspace(-10, 10, 1000)
for i, color in zip(range(0, 150, 15), plt.cm.Greys(np.linspace(.2, 1, 10))):

    # extract the sufficient statistics from the input node (and parents)
    mean = df.x_1_expected_mean.iloc[i]
    std = np.sqrt(
        1/(mean_precision_hgf.attributes[0]["expected_precision"] * (1/np.exp(df.x_2_expected_mean.iloc[i])))
)

    # the model expectations
    ax.plot(x, norm(mean, std).pdf(x), color=color, label=i)


# the sampling distribution
ax.fill_between(x, norm(dist_mean, dist_std).pdf(x), color="#582766", alpha=.2)

ax.legend(title="Iterations")
ax.set_xlabel("Input (u)")
ax.set_ylabel("Density")
plt.grid(linestyle=":")
sns.despine()
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

## System configuration

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
%load_ext watermark
%watermark -n -u -v -iv -w -p pyhgf,jax,jaxlib
```

```{code-cell} ipython3

```
