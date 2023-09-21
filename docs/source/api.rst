.. _api_ref:

.. currentmodule:: pyhgf


.. contents:: Table of Contents
   :depth: 2

API
+++

Updates functions
-----------------

The `updates` module contains the update function used both for prediction and prediction-error steps during the belief propagation. These functions call intenaly the more specific update function listed in the prediction and prediction-error sub-modules.

Updating binary nodes
=====================

Core functionnalities to update *binary* nodes.

.. currentmodule:: pyhgf.updates.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.binary

    binary_node_prediction
    binary_input_prediction
    binary_node_prediction_error
    binary_input_prediction_error

Updating continuous nodes
=========================

Core functionnalities to update *continuous* nodes.

.. currentmodule:: pyhgf.updates.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.continuous

    continuous_node_prediction_error
    continuous_node_prediction
    continuous_input_prediction_error
    continuous_input_prediction

Updating categorical nodes
==========================

Core functionnalities to update *categorical* nodes.

.. currentmodule:: pyhgf.updates.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.categorical

    categorical_input_update

Prediction error steps
======================

Propagate prediction errors to the value and volatility parents of a given node.

Continuous nodes
~~~~~~~~~~~~~~~~

.. currentmodule:: pyhgf.updates.prediction_error.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.continuous

    prediction_error_mean_value_parent
    prediction_error_precision_value_parent
    prediction_error_value_parent
    prediction_error_volatility_parent
    prediction_error_precision_volatility_parent
    prediction_error_mean_volatility_parent
    prediction_error_input_value_parent
    prediction_error_input_mean_value_parent
    prediction_error_input_precision_value_parent

Binary nodes
~~~~~~~~~~~~

.. currentmodule:: pyhgf.updates.prediction_error.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.binary

    prediction_error_mean_value_parent
    prediction_error_precision_value_parent
    prediction_error_value_parent
    prediction_error_input_value_parent
    input_surprise_inf
    input_surprise_reg

Prediction steps
================

Compute the expectation for future observation given the influence of parent nodes.

Continuous nodes
~~~~~~~~~~~~~~~~

.. currentmodule:: pyhgf.updates.prediction.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.continuous

    prediction_mean_value_parent
    prediction_precision_value_parent
    prediction_value_parent
    prediction_precision_volatility_parent
    prediction_mean_volatility_parent

Binary nodes
~~~~~~~~~~~~

.. currentmodule:: pyhgf.updates.prediction.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.binary

    prediction_input_value_parent


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

Math
----

Math functions and probability densities.

.. currentmodule:: pyhgf.math

.. autosummary::
   :toctree: generated/pyhgf.math

    gaussian_density
    sigmoid
    binary_surprise
    gaussian_surprise
    dirichlet_kullback_leibler