---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

(probabilistic_networks)=
# Creating and manipulating networks of probabilistic nodes

[pyhgf](https://ilabcode.github.io/pyhgf/index.html#) is designed with inspiration from graph neural network libraries that can support message-passing schemes and performs beliefs propagation through network nodes. Here, this principle is applied to predictive processing modelisation and focuses on networks that are structured as rooted trees and perform variational message passing to update beliefs about the state of the environment, inferred from the observations that are provided at the root of the tree. While this library is well-suited and optimized to implement the standard two-level and three-level HGF, it can also be applied to much larger use cases and the main idea is to generalize belief propagation as it has been described so far to larger and more complex networks that will capture greater variety of environmental structure. Therefore, the library is also designed to facilitate the creation and manipulation of such probabilistic networks. The main idea is that a network can be fully described by the following variables:
1. the network parameters
2. the network structure
3. the update function(s)
4. the update sequence(s)
Splitting the networks this way makes the components easily compatible with JAX main transformations, and dynamically accessible during the inference processes, allowing the creation of agents that can manipulate these components to minimize surprise. 

In this notebook, we dive into the details of creating such networks and illustrate their modularity by the manipulation of the four main variables.

+++

## Theory and implementation details

Any probabilistic network can be fully described by the following variables. Let 

$$\mathcal{HGF}_{k} = \{\theta, \xi, \mathcal{F}, \Sigma \}$$ 

be a probabilistic network that contains $K$ probabilistic nodes. The variable 

$$\theta = \{\theta_1, ..., \theta_{k}\}$$ 

is the set of parameters, and each parameter is a set of values. Nodes' parameters can be used to register sufficient statistics of the distributions as well as various coupling weights. This component is registered as the `parameters_structure` dictionary.

The *shape* of the hierarchical structure is defined by its [adjacency list](https://en.wikipedia.org/wiki/Adjacency_list), which register the connections between a node and other nodes from the network.

$$\xi = \{\xi_1, ..., \xi_{k} \}$$

where every edge $\xi_k$ contains $m$ sets of node indexes, $m$ being the adjacency dimension (here we only consider value and volatility coupling, therefore $m=2$). 

The way beliefs are being updated, or the type of generic computation that is performed by the nodes are defined by the set of $n$ update functions

$$\mathcal{F} = \{f_1, ..., f_n\}$$

In this set, each update function is linked to a node from the node indexes $n \in 1, ..., k$ to which the function should apply. The most standard uses of the HGF only require continuous and/or binary update functions for input and states node that can be found in the {ref}`pyhgf.continuous` and {ref}`pyhgf.binary` sub-modules.

The dynamic of belief propagation dynamics (which part of the network should be updated and in which order) are controlled by the ordered update sequence

$$\Sigma = [f_1(n_1), ..., f_i(n_j), f \in \mathcal{F}, n \in 1, ..., k ]$$,

This list describes the sequence of function-to-nodes instructions that are executed during the inference and update processes.

![graph_networks](../images/graph_networks.svg)

```{tip} Compatibility with JAX transformations
One of the advantages of reasoning this way is that it dissociates variables that are transparent to the JAX framework and can be expressed as "PyTress" from variables that should be filtered before transformations. The variable `parameters_structure` ($\theta$) is typically expressed as a PyTree while the other variables that contain parametrized functions are filtered. See [the documattion](https://jax.readthedocs.io/en/latest/notebooks/thinking_in_jax.html#jit-mechanics-tracing-and-static-variables) for further details on JAX transformations.
```

+++

### Creating probabilistic nodes

```{code-cell} ipython3
from pyhgf.typing import Indexes
parameters = {"mu": 0.0, "pi": 1.0}

parameters_structure = (parameters, parameters, parameters)
node_structure = (
    Indexes((1,), None, None, None),
    Indexes(None, (2,), (0,), None),
    Indexes(None, None, None, (1,)),
)
```

The code above illustrate the creation of a probabilistic network of 3 nodes with simple parameters sets $(\mu=0.0, \pi=1.0)$. Node 2 is the value parent of node 1. Node 3 is the value parent of node 2 and has no parents.

+++

### Visualizing probabilistic network

```{code-cell} ipython3
from pyhgf.model import HGF

# create a three-level HGF using default parameters
hgf = HGF(n_levels=3, model_type="continuous")
hgf.plot_network()
```

## Modifying the parameter structure

The simpler change we can make on a network is to change the values of some of its parameters. The parameters are stored in the `parameters_structure` variable as a dictionary where the key (integers) are node indexes. Therefore, modifying the expected precision of the third node in the previous example is as simple as:

```{code-cell} ipython3
hgf.parameters_structure[3]["pi"] = 5.0
```

However, modifying parameters values *manually* should not be that common as this is something we want the model to perform dynamically as we present new observations, but this can be used for example to generate prior predictive by sampling some parameter values from a distribution. 

```{note} What is a valid parameter/value?
A probabilistic node can store an arbitrary number of parameters. Parameter values should be valid JAX types, therefore a node cannot contain strings. You can provide additional parameters by using the `additional_parameters` arguments in {py:meth}`pyhgf.model.add_input_node`, {py:meth}`pyhgf.model.add_value_parent` and {py:meth}`pyhgf.model.add_volatility_parent`. Most of the nodes that are being used in the HGF use Gaussian distribution, therefore they contain the current mean and precision (`mu` and `pi`) as well as the expected mean and precision (`muhat` and `pihat`).
```

+++

## Modifying the network structure

The second way we can modify a probabilistic network is by modifying its structure (i.e. the number of nodes, the type of nodes and the way they are connected with each other). Because nodes and connections are modular, a large variety of network structures can be declared. The only restrictions are that the network should be **acyclic** and **rooted** (the roots being input nodes).

For example, the following networks are valid HGF structures:

```{code-cell} ipython3
custom_hgf = (
    HGF(model_type=None)
    .add_input_node(kind="continuous")
    .add_value_parent(children_idxs=[0])
    .add_value_parent(children_idxs=[1])
    .add_value_parent(children_idxs=[1])
    .add_value_parent(children_idxs=[1])
    .add_volatility_parent(children_idxs=[2, 3])
    .add_volatility_parent(children_idxs=[4])
    .add_value_parent(children_idxs=[5, 6])
)
custom_hgf.plot_network()
```

```{code-cell} ipython3
custom_hgf = (
    HGF(model_type=None)
    .add_input_node(kind="continuous", input_idx=0)
    .add_input_node(kind="continuous", input_idx=1)
    .add_input_node(kind="continuous", input_idx=2)
    .add_value_parent(children_idxs=[0])
    .add_value_parent(children_idxs=[1])
    .add_value_parent(children_idxs=[2])
    .add_volatility_parent(children_idxs=[3, 4, 5])
)
custom_hgf.plot_network()
```

The structure of the probabilistic network is stored in the `node_structure` variable which consists of a tuple of `Indexes` that store the indexes of value/volatility parents/children for each node. For example, accessing the nodes connected to node `4` in the example above is done with:

```{code-cell} ipython3
# the node structure
custom_hgf.node_structure[4]
```

```{tip} Different types of coupling
Contrary to standard graph networks, the directed connection between nodes can have multiple forms. The standard HGF is built on combinations of *value coupling* and *volatility coupling* bindings, but this can be extended to an arbitrary number of types.
```

+++

### Multivariate coupling

As we can see in the examples above, nodes in a valid HGF network can be influenced by multiple parents (either value or volatility parents). Similarly, a single node can be influenced by multiple children. This feature is termed *multivariate descendency* and *multivariate ascendency* (respectively) and is a central addition to the generalization of the HGF {cite:p}`weber:2023` that was implemented in this package, as well as in the [Julia counterpart](https://github.com/ilabcode/HierarchicalGaussianFiltering.jl).

+++

```{note}
Hierarchical Gaussian Filters have often been described in terms of levels. For example, the two-level and three-level HGFs are specific instances of  node structures designed to track the volatility and meta-volatility of binary or continuous processes, respectively. While such models can easily be implemented in this framework, the notion of level itself is now less informative. This is a consequence of the multivariate control and influence properties, which can result in structures in which some nodes have a position in the hierarchy that cannot be clearly disambiguated.
```

+++

#### The case of *multivariate descendency*

*Multivariate descendency* refers to situations where a node exerts a predictive influence (and is updated backwards through prediction errors) on multiple children nodes, either via value or volatility coupling.

##### Value coupling

```{code-cell} ipython3
import numpy as np
```

```{code-cell} ipython3
# simulate some time series - sampling from N(0, 1) sometime shifted by 5
u_0 = np.random.normal(0, 1, size=100)
u_1 = np.random.normal(0, 1, size=100)

u_0[20:50] += 5
u_0[90:] += 5

u_1[50:70] += 5
u_1[70:] += 5

input_data = np.array([u_0, u_1]).T
```

```{code-cell} ipython3
# creating a network that contains many child nodes are value coupled to one parent node
many_value_children_hgf = (
    HGF(model_type=None)
    .add_input_node(kind="continuous")
    .add_input_node(kind="continuous", input_idx=1)
    .add_value_parent(children_idxs=[0, 1])
    .add_volatility_parent(children_idxs=[2])
)

# plot the network
many_value_children_hgf.plot_network()
```

```{code-cell} ipython3
many_value_children_hgf.input_data(input_data=input_data)
```

```{code-cell} ipython3
# many_children_hgf.plot_nodes([0, 1, 2], figsize=(12, 9))
```

```{code-cell} ipython3
#plt.plot(many_children_hgf.to_pandas().observation_input_0, label="Input-0")
#plt.plot(many_children_hgf.to_pandas().observation_input_1, label="Input-1")
#plt.plot(many_children_hgf.to_pandas().x_2_mu, label="mu_x2")
#plt.plot(many_children_hgf.to_pandas().x_3_mu, label="mu_x3")
#plt.legend()
```

##### Volatility coupling

```{code-cell} ipython3
# simulate some time series - sampling from N(0, 1) sometime multiplied by 5
u_0 = np.random.normal(0, 1, size=1000)
u_1 = np.random.normal(0, 1, size=1000)

u_0[200:500] *= 5
u_0[900:] *= 5

u_1[800:] *= 5

input_data = np.array([u_0, u_1]).T
```

```{code-cell} ipython3
# creating a network that contains many child nodes are volatility coupled to one parent node
many_volatility_children_hgf = (
    HGF(model_type=None)
    .add_input_node(kind="continuous")
    .add_input_node(kind="continuous", input_idx=1)
    .add_value_parent(children_idxs=[0])
    .add_value_parent(children_idxs=[1])
    .add_volatility_parent(children_idxs=[2, 3])
)

# plot the network
many_volatility_children_hgf.plot_network()
```

```{code-cell} ipython3
many_volatility_children_hgf.input_data(input_data=input_data)
```

```{code-cell} ipython3
# many_volatility_children_hgf.plot_nodes([2, 3, 4])
```

```{code-cell} ipython3
#plt.plot(many_volatility_children_hgf.to_pandas().observation_input_0, label="Input-0")
#plt.plot(many_volatility_children_hgf.to_pandas().observation_input_1, label="Input-1")
#plt.legend()
```

```{code-cell} ipython3
#plt.plot(many_volatility_children_hgf.to_pandas().x_2_mu, label="mu_x2")
#plt.plot(many_volatility_children_hgf.to_pandas().x_3_mu, label="mu_x3")
#plt.legend()
```

```{code-cell} ipython3
#plt.plot(many_volatility_children_hgf.to_pandas().x_4_pi, label="pi_x4")
```

```{code-cell} ipython3
#plt.plot(many_volatility_children_hgf.to_pandas().x_4_mu, label="mu_x4")
```

#### The case of *multivariate ascendency*

*Multivariate ascendency* refers to situations where a node is influenced (and updates backwards through prediction errors) multiple value or volatility parents.

#### Value coupling

```{warning}
Work in progress
```

#### Volatility coupling

```{warning}
Work in progress
```

+++

## Creating custom update functions

Models that are created with parameter structures and node structures only represent static configuration of the HGF, they are not yet filtering new observation or acting in response to specific input. To add a dynamic and responsive layer, we need two additional components:

- update functions $\mathcal{F} = \{f_1, ..., f_n\}$
- update sequences $\Sigma = [f_1(n_1), ..., f_i(n_j), f \in \mathcal{F}, n \in 1, ..., k ]$,


### Update functions

Update functions are the heart of the HGF filtering procedure, these functions implement the message passing and parameters updating steps between node. An update function in its simplier form is a fuction defined as

```python
parameters_structure = update_fn(node_idx, parameters_structure, node_structure)
```

In other words, it is updating the parameters structure by applying certain transformation using the node $i$ as reference. This usually means that an observation has reached node $i$ and we want to send prediction-error to the parent nodes and update their sufficient statistics. The function has acess to the entire parameters and nodes structure, which mean that it can retrive parameters from parents, children, grand-parents etc... But in practice, pessage passing updates only use information found in the [Markov blanket](https://en.wikipedia.org/wiki/Markov_blanket) of the given node.

+++

## Creating custom update sequences

Update sequences define the dynamics of beliefs propagation through the probabilistic network. In its simpler form, an update sequence is a sequence of update functions pointing to a node index such as:

```python
update_sequence = (
    (update_fn1, 0),
    (update_fn2, 1),
)
```

The HGF class include a built-in {ref}`pyhgf.modeal.HGF.get_update_sequence` method to automatically generate the update sequence from the network structure, assuming that we want to propagate the beliefs from the lower part of the tree (the input nodes) to its upper part (nodes that do not have parents).

+++

## Time-varying update sequences

### Static assignation of update sequences

### Dynamic assignation of update sequences

```{warning}
Work in progress
```

```{code-cell} ipython3

```
