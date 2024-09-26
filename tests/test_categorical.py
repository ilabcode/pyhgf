# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import jax.numpy as jnp
import numpy as np

from pyhgf.model import Network


def test_categorical_state_node():
    # generate some categorical inputs data
    np.random.seed(123)
    input_data = np.array(
        [np.random.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(3)]
    ).T
    input_data = np.vstack([[0.0] * input_data.shape[1], input_data])

    # create the categorical HGF
    categorical_hgf = Network().add_nodes(
        kind="categorical-state",
        node_parameters={
            "n_categories": 3,
            "binary_parameters": {"tonic_volatility_2": -2.0},
        },
    )

    # fitting the model forwards
    categorical_hgf.input_data(
        input_data=(input_data, np.ones(input_data.shape, dtype=int))
    )

    # export to pandas data frame
    categorical_hgf.to_pandas()

    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["kl_divergence"].sum(), 0.8222526
    )
    assert jnp.isclose(
        categorical_hgf.node_trajectories[0]["surprise"].sum(), 7.6738853
    )
