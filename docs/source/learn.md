# Learn

```{toctree}
---
hidden:
glob:
---

.notebooks/*.md

```

## Theory

::::{grid} 2

:::{grid-item-card}  Creating and manipulating networks of probabilistic nodes
:link: probabilistic_networks
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/networks.png

How to create and manipulate a network of probabilistic nodes for reinforcement learning? Working at the intersection of graphs, neural networks and probabilistic frameworks.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/0-Creating_networks.ipynb)

:::

:::{grid-item-card}  An introduction to the Hierarchical Gaussian Filter
:link: theory
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/hgf.png


How the generative model of the Hierarchical Gaussian filter can be turned into update functions that update nodes through value and volatility coupling?

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/0-Theory.ipynb)
:::
::::


## Tutorials

::::{grid} 3

:::{grid-item-card}  The binary Hierarchical Gaussian Filter
:link: binary_hgf
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/binary.png

Introducing with example the binary Hierarchical Gaussian filter and its applications to reinforcement learning.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/1.1-Binary_HGF.ipynb)
:::

:::{grid-item-card}  The continuous Hierarchical Gaussian Filter
:link: continuous_hgf
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/hgf.png


Introducing with example the continuous Hierarchical Gaussian filter and its applications to signal processing.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/1.2-Continuous_HGF.ipynb)
:::

:::{grid-item-card}  The categorical Hierarchical Gaussian Filter
:link: categorical_hgf
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/binary.png


Introducing the categorical Hierarchical Gaussian Filter as a generalisation of the binary version.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/1.´3-CAtegorical_HGF.ipynb)
:::

:::

:::{grid-item-card}  Using custom response functions
:link: custom_response_functions
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/binary.png


How to adapt any model to specific behaviours and experimental design by using custom response functions.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/2-Using_custom_response_functions.ipynb)
:::

:::

:::{grid-item-card}  Embedding the Hierarchical Gaussian Filter in a Bayesian network for multilevel inference
:link: multilevel_hgf
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/binary.png


How to use any model as a distribution to perform hierarchical inference at the group level.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/3-Multilevel_HGF.ipynb)
:::

:::{grid-item-card}  Embedding the Hierarchical Gaussian Filter in a Bayesian network for multilevel inference
:link: parameters_recovery
:link-type: ref
:img-top: https://github.com/ilabcode/pyhgf/blob/master/docs/source/images/binary.png


How to use any model as a distribution to perform hierarchical inference at the group level.

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/4-Parameter_recovery.ipynb)
:::
::::

## Use cases

::::{grid} 3

:::{grid-item-card}  Bayesian filtering of cardiac dynamics
:link: example_1
:link-type: ref

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/Example_1_Heart_rate_variability.ipynb)

:::

:::{grid-item-card}  Value and volatility coupling with an input node
:link: example_2
:link-type: ref

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/Example_2_Input_node_volatility_coupling.ipynb)
:::
::::

## Exercises

Hand-on exercises to build intuition around the main components of the HGF and use an agent that optimizes its action under noisy observations.


::::{grid} 3

:::{grid-item-card}  Bayesian filtering of cardiac dynamics
:link: hgf_exercises
:link-type: ref

+++
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilabcode/pyhgf/blob/master/docs/source/notebooks/Exercise_1_Using_the_HGF.ipynb)

:::
::::
