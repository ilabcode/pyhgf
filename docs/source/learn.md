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
:img-top: ./images/graph_networks.svg

How to create and manipulate a network of probabilistic nodes for reinforcement learning? Working at the intersection of graphs, neural networks and probabilistic frameworks.

:::

:::{grid-item-card}  An introduction to the Hierarchical Gaussian Filter
:link: theory
:link-type: ref
:img-top: ./images/trajectories.png


How the generative model of the Hierarchical Gaussian filter can be turned into update functions that update nodes through value and volatility coupling?

:::
::::


## Tutorials

::::{grid} 3

:::{grid-item-card}  The binary Hierarchical Gaussian Filter
:link: binary_hgf
:link-type: ref
:img-top: ./images/binary.png

Introducing with example the binary Hierarchical Gaussian filter and its applications to reinforcement learning.

:::

:::{grid-item-card}  The categorical Hierarchical Gaussian Filter
:link: categorical_hgf
:link-type: ref
:img-top: ./images/categorical.png

The categorical Hierarchical Gaussian Filter as a generalisation of the binary version.

:::

:::{grid-item-card}  The continuous Hierarchical Gaussian Filter
:link: continuous_hgf
:link-type: ref
:img-top: ./images/continuous.png


Introducing with example the continuous Hierarchical Gaussian filter and its applications to signal processing.

+++

:::
::::


::::{grid} 3

:::{grid-item-card}  Using custom response functions
:link: custom_response_functions
:link-type: ref
:img-top: ./images/response_models.png


How to adapt any model to specific behaviours and experimental design by using custom response functions.

:::

:::{grid-item-card}  Embedding the Hierarchical Gaussian Filter in a Bayesian network for multilevel inference
:link: multilevel_hgf
:link-type: ref
:img-top: ./images/response_models.png


How to use any model as a distribution to perform hierarchical inference at the group level.

:::

:::{grid-item-card}  Embedding the Hierarchical Gaussian Filter in a Bayesian network for multilevel inference
:link: parameters_recovery
:link-type: ref
:img-top: ./images/binary.png


How to use any model as a distribution to perform hierarchical inference at the group level.

:::
::::

## Use cases

::::{grid} 3

:::{grid-item-card}  Bayesian filtering of cardiac dynamics
:link: example_1
:link-type: ref

:::

:::{grid-item-card}  Value and volatility coupling with an input node
:link: example_2
:link-type: ref
:img-top: ./images/input_mean_precision.png

:::
::::

## Exercises

Hand-on exercises to build intuition around the main components of the HGF and use an agent that optimizes its action under noisy observations.


::::{grid} 3

:::{grid-item-card}  Applying the Hierarchical Gaussian Filter through practical exercises
:link: hgf_exercises
:link-type: ref

:::
::::
