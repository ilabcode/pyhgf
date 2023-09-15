.. _api_ref:

.. currentmodule:: pyhgf


.. contents:: Table of Contents
   :depth: 2

API
+++

Nodes updates
-------------

Update functions used to update the posteriors or predict outcomes of probabilistic nodes. These functions call intenaly the more specific update function listed in the posterior and prediction sections below.

Binary
******

Core functionnalities to update *binary* nodes.

.. currentmodule:: pyhgf.updates.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.binary

    binary_node_update
    binary_input_update
    gaussian_density
    sgm
    binary_surprise


Continuous
**********

Core functionnalities to update *continuous* nodes.

.. currentmodule:: pyhgf.updates.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.continuous

    continuous_node_update
    continuous_node_prediction
    continuous_input_update
    continuous_input_prediction
    gaussian_surprise

Categorical
***********

Core functionnalities to update *categorical* nodes.

.. currentmodule:: pyhgf.updates.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.categorical

    categorical_input_update
    dirichlet_kullback_leibler

Posterior updates
-----------------

Continuous
::::::::::

.. currentmodule:: pyhgf.updates.posterior.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous

    update_mean_value_parent
    update_precision_value_parent
    update_value_parent
    update_volatility_parent
    update_precision_volatility_parent
    update_mean_volatility_parent

Prediction updates
------------------

Continuous
::::::::::


Distribution
------------

The Herarchical Gaussian Filter as a PyMC distribution. This distribution can be
embedded in models using PyMC>=5.0.0.

.. currentmodule:: pyhgf.distribution

.. autosummary::
   :toctree: generated/pyhgf.distribution

   hgf_logp
   HGFLogpGradOp
   HGFDistribution

Model
-----

The main class used to create a standard Hierarchical Gaussian Filter for binary or
continuous inputs, with two or three levels. This class wraps the previous JAX modules
and create a standard node structure for these models.

.. currentmodule:: pyhgf.model

.. autosummary::
   :toctree: generated/pyhgf.model

   HGF

Plots
-----

Plotting functionnalities to visualize parameters trajectories and correlations after
observing new data.

.. currentmodule:: pyhgf.plots

.. autosummary::
   :toctree: generated/pyhgf.plots

   plot_trajectories
   plot_correlations
   plot_network
   plot_nodes

Response
--------

A collection of responses functions. A response function is simply a callable taking at
least the HGF instance as input after observation and returning surprise.

.. currentmodule:: pyhgf.response

.. autosummary::
   :toctree: generated/pyhgf.response

   first_level_gaussian_surprise
   total_gaussian_surprise
   first_level_binary_surprise

Networks
--------

Utilities for manipulating networks of probabilistic nodes.

.. currentmodule:: pyhgf.networks

.. autosummary::
   :toctree: generated/pyhgf.networks

   beliefs_propagation
   trim_sequence
   list_branches
   fill_categorical_state_node
   get_update_sequence
