.. _api_ref:

.. currentmodule:: ghgf


.. contents:: Table of Contents
   :depth: 2

API
+++

Binary
------

Core functionnalities of the JAX implementation to update *binary* nodes.

.. currentmodule:: ghgf.binary

.. autosummary::
   :toctree: generated/ghgf.binary

    binary_node_update
    binary_input_update
    gaussian_density
    sgm
    binary_surprise
    loop_binary_inputs


Continuous
----------

Core functionnalities of the JAX implementation to update *continuous* nodes.

.. currentmodule:: ghgf.continuous

.. autosummary::
   :toctree: generated/ghgf.continuous

    continuous_node_update
    continuous_input_update
    gaussian_surprise
    loop_continuous_inputs

Model
-----

The main class used to create a standard Hierarchical Gaussian Filter for binary or
continuous inputs, with two or three levels. This class wraps the previous JAX modules
and create a standard node structure for these models.

.. currentmodule:: ghgf.model

.. autosummary::
   :toctree: generated/ghgf.model

   HGF

Plots
-----

Plotting functionnalities to visualize parameters trajectories and correlations after
observing new data.

.. currentmodule:: ghgf.plots

.. autosummary::
   :toctree: generated/ghgf.plots

   plot_trajectories
   plot_correlations


PyMC
----

The Herarchical Gaussian Filter as a PyMC distribution. This distribution can be
embedded in a PyMC model and the free paramters will be estimated e.g. using MCMC
sampling.

.. currentmodule:: ghgf.pymc

.. autosummary::
   :toctree: generated/ghgf.pymc

   hgf_logp
   HGFLogpGradOp
   HGFDistribution


Response
--------

A collection of responses functions. A response function is simply a callable taking at
least the HGF instance as input after observation and returning surprise.

.. currentmodule:: ghgf.response

.. autosummary::
   :toctree: generated/ghgf.response

   gaussian_surprise
   binary_surprise

Typing
------

Types hints.

.. currentmodule:: ghgf.typing

.. autosummary::
   :toctree: generated/ghgf.typing

   node_validation

Python
------

The pure Python implementation of the Hierarchical Gaussian Filter for binary or
continuous inputs, with two or three levels. This implementation is independent from the
more recent JAX implementation and is not required to run the code. It is currently used
for testing.

.. currentmodule:: ghgf.python

.. autosummary::
   :toctree: generated/ghgf.python

   Model
   StandardHGF
   StandardBinaryHGF
   StateNode
   BinaryNode
   InputNode
   BinaryInputNode
   Parameter
   exp_shift_mirror
   log_shift_mirror
   sgm
   logit
   gaussian
   gaussian_surprise
   binary_surprise
