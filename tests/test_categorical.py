# Author: Nicolas Legrand <nicolas.legrand@cas.au.dk>

import numpy as np

from pyhgf.model import Network


def test_categorical_state_node():
    # generate some categorical inputs data
    input_data = np.array(
        [np.random.multinomial(n=1, pvals=[0.1, 0.2, 0.7]) for _ in range(3)]
    ).T
    input_data = np.vstack([[0.0] * input_data.shape[1], input_data])

    # create the categorical HGF
    categorical_hgf = Network().add_nodes(
        kind="categorical-input",
        node_parameters={
            "n_categories": 3,
            "binary_parameters": {"tonic_volatility_2": -2.0},
        },
    )

    # fitting the model forwards
    categorical_hgf.input_data(input_data=input_data.T)

    # export to pandas data frame
    categorical_hgf.to_pandas()
