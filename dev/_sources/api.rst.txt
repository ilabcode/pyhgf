.. _api_ref:

.. currentmodule:: pyhgf


.. contents:: Table of Contents
   :depth: 5

API
###

Updates functions
*****************

Update functions are the heart of probabilistic networks as they shape the propagation of beliefs in the neural hierarchy. The library implements the standard variational updates for value and volatility coupling, as described in Weber et al. (2023).

The `updates` module contains the update functions used during the belief propagation. Update functions are available through three sub-modules, organized according to their functional roles. We usually dissociate the first updates, triggered top-down (from the leaves to the roots of the networks), that are prediction steps and recover the current state of inference. The second updates are the prediction error, signalling the divergence between the prediction and the new observation (for input nodes), or state (for state nodes). Interleaved with these steps are posterior update steps, where a node receives prediction errors from the child nodes and estimates new statistics.


Posterior updates
=================

Update the sufficient statistics of a state node after receiving prediction errors from children nodes. The prediction errors from all the children below the node should be computed before calling the posterior update step.

Binary nodes
------------

.. currentmodule:: pyhgf.updates.posterior.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.binary

    binary_node_update_infinite
    binary_node_update_finite


Categorical nodes
-----------------

.. currentmodule:: pyhgf.updates.posterior.categorical

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.categorical

    categorical_input_update

Continuous nodes
----------------

.. currentmodule:: pyhgf.updates.posterior.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.continuous

    posterior_update_mean_continuous_node
    posterior_update_precision_continuous_node
    continuous_node_update
    continuous_node_update_ehgf

Exponential family
------------------

.. currentmodule:: pyhgf.updates.posterior.exponential

.. autosummary::
   :toctree: generated/pyhgf.updates.posterior.exponential

    posterior_update_exponential_family

Prediction steps
================

Compute the expectation for future observation given the influence of parent nodes. The prediction step are executed for all nodes, top-down, before any observation.

Binary nodes
------------

.. currentmodule:: pyhgf.updates.prediction.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.binary

    binary_state_node_prediction

Continuous nodes
----------------

.. currentmodule:: pyhgf.updates.prediction.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.continuous

    predict_mean
    predict_precision
    continuous_node_prediction

Dirichlet processes
-------------------

.. currentmodule:: pyhgf.updates.prediction.dirichlet

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction.dirichlet

    dirichlet_node_prediction

Prediction error steps
======================

Compute the value and volatility prediction errors of a given node. The prediction error can only be computed after the posterior update (or observation) of a given node.

Inputs
------

Binary inputs
^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.inputs.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.inputs.binary

    binary_input_prediction_error_infinite_precision
    binary_input_prediction_error_finite_precision

Continuous inputs
^^^^^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.inputs.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.inputs.continuous

    continuous_input_volatility_prediction_error
    continuous_input_value_prediction_error
    continuous_input_prediction_error

Generic input
^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.inputs.generic

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.inputs.generic

    generic_input_prediction_error

State nodes
-----------


Binary state nodes
^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.nodes.binary

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.nodes.binary

    binary_state_node_prediction_error

Continuous state nodes
^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.nodes.continuous

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.nodes.continuous

    continuous_node_value_prediction_error
    continuous_node_volatility_prediction_error
    continuous_node_prediction_error

Dirichlet processes
^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pyhgf.updates.prediction_error.nodes.dirichlet

.. autosummary::
   :toctree: generated/pyhgf.updates.prediction_error.nodes.dirichlet

    dirichlet_node_prediction_error
    update_cluster
    create_cluster
    get_candidate
    likely_cluster_proposal
    clusters_likelihood

Distribution
************

The Hierarchical Gaussian Filter as a PyMC distribution. This distribution can be
embedded in models using PyMC>=5.0.0.

.. currentmodule:: pyhgf.distribution

.. autosummary::
   :toctree: generated/pyhgf.distribution

   hgf_logp
   HGFLogpGradOp
   HGFDistribution

Model
*****

The main class is used to create a standard Hierarchical Gaussian Filter for binary or
continuous inputs, with two or three levels. This class wraps the previous JAX modules
and creates a standard node structure for these models.

.. currentmodule:: pyhgf.model

.. autosummary::
   :toctree: generated/pyhgf.model

   HGF

Plots
*****

Plotting functionalities to visualize parameters trajectories and correlations after
observing new data.

.. currentmodule:: pyhgf.plots

.. autosummary::
   :toctree: generated/pyhgf.plots

   plot_trajectories
   plot_correlations
   plot_network
   plot_nodes

Response
********

A collection of response functions. A response function is simply a callable taking at
least the HGF instance as input after observation and returning surprise.

.. currentmodule:: pyhgf.response

.. autosummary::
   :toctree: generated/pyhgf.response

   first_level_gaussian_surprise
   total_gaussian_surprise
   first_level_binary_surprise
   binary_softmax
   binary_softmax_inverse_temperature

Utils
*****

Utilities for manipulating neural networks.

.. currentmodule:: pyhgf.utils

.. autosummary::
   :toctree: generated/pyhgf.utils

   beliefs_propagation
   trim_sequence
   list_branches
   fill_categorical_state_node
   get_update_sequence
   concatenate_networks
   add_edges

Math
****

Math functions and probability densities.

.. currentmodule:: pyhgf.math

.. autosummary::
   :toctree: generated/pyhgf.math

    MultivariateNormal
    Normal
    gaussian_predictive_distribution
    gaussian_density
    sigmoid
    binary_surprise
    gaussian_surprise
    dirichlet_kullback_leibler
    binary_surprise_finite_precision